#pragma once
#include <onnxruntime_cxx_api.h>

#include <array>
#include <cmath>
#include <string>
#include <vector>

#include "dsp/fft.hpp"   // reused from DSP-Fundamentals-Toolkit
#include "dsp/stft.hpp"  // hann_window

namespace rt {
constexpr size_t kFrame = 512;
constexpr size_t kHop = 128;
constexpr size_t kBins = kFrame / 2 + 1;   // 257
// Streaming denoiser: feed kHop new samples at a time, get kHop out.
// Latency = one frame (32 ms) + hop buffering. No allocation after ctor.
// The recurrent/CNN state is a single opaque tensor whose shape is read from
// the ONNX model at load time, so the SAME engine runs any streamable model
// (GRU/RNN state (2,1,256), LSTM (4,1,256), CNN (1,context,257), ...).
class Denoiser {
public:
    explicit Denoiser(const std::string& model_path)
        : env_(ORT_LOGGING_LEVEL_WARNING, "denoiser"),
          session_(env_, model_path.c_str(), Ort::SessionOptions{}),
          window_(dsp::hann_window(kFrame)),
          in_buf_(kFrame, 0.0f),
          ola_buf_(kFrame, 0.0f),
          norm_buf_(kFrame, 0.0f),
          frame_(kFrame),
          log_mag_(kBins),
          mem_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
        discover_state_shape();
    }
    // in/out: exactly kHop samples each.
    void process_hop(const float* in, float* out) {
        // 1. Slide input buffer, append new hop
        std::copy(in_buf_.begin() + kHop, in_buf_.end(), in_buf_.begin());
        std::copy(in, in + kHop, in_buf_.end() - kHop);
        // 2. Windowed FFT of current frame
        for (size_t i = 0; i < kFrame; ++i)
            frame_[i] = dsp::cpx(in_buf_[i] * window_[i], 0.0f);
        dsp::fft(frame_);
        // 3. log1p magnitude -> model -> mask
        for (size_t k = 0; k < kBins; ++k)
            log_mag_[k] = std::log1p(std::abs(frame_[k]));
        run_model();
        // 4. Apply mask (keep phase), rebuild full spectrum
        for (size_t k = 0; k < kBins; ++k) frame_[k] *= mask_[k];
        for (size_t k = kBins; k < kFrame; ++k)
            frame_[k] = std::conj(frame_[kFrame - k]);
        dsp::fft(frame_, /*inverse=*/true);
        // 5. Overlap-add with window^2 normalization
        std::copy(ola_buf_.begin() + kHop, ola_buf_.end(), ola_buf_.begin());
        std::fill(ola_buf_.end() - kHop, ola_buf_.end(), 0.0f);
        std::copy(norm_buf_.begin() + kHop, norm_buf_.end(), norm_buf_.begin());
        std::fill(norm_buf_.end() - kHop, norm_buf_.end(), 0.0f);
        for (size_t i = 0; i < kFrame; ++i) {
            ola_buf_[i]  += frame_[i].real() * window_[i];
            norm_buf_[i] += window_[i] * window_[i];
        }
        // 6. Oldest kHop samples are now complete -> output
        for (size_t i = 0; i < kHop; ++i)
            out[i] = ola_buf_[i] / (norm_buf_[i] + 1e-8f);
    }
    void reset_state() { std::fill(state_.begin(), state_.end(), 0.0f); }

private:    // Read the "state_in" tensor shape from the model and size the state buffer.
    void discover_state_shape() {
        Ort::AllocatorWithDefaultOptions alloc;
        for (size_t i = 0; i < session_.GetInputCount(); ++i) {
            auto name = session_.GetInputNameAllocated(i, alloc);
            if (std::string(name.get()) == "state_in") {
                state_dims_ = session_.GetInputTypeInfo(i)
                                  .GetTensorTypeAndShapeInfo()
                                  .GetShape();
                break;
            }
        }
        size_t total = 1;
        for (int64_t d : state_dims_) total *= size_t(d);
        state_.assign(total, 0.0f);
    }
    void run_model() {
        const std::array<int64_t, 3> frame_shape{1, 1, int64_t(kBins)};

        Ort::Value inputs[2] = {
            Ort::Value::CreateTensor<float>(mem_info_, log_mag_.data(),
                                            log_mag_.size(), frame_shape.data(), 3),
            Ort::Value::CreateTensor<float>(mem_info_, state_.data(),
                                            state_.size(), state_dims_.data(),
                                            state_dims_.size())};

        const char* in_names[] = {"log_mag", "state_in"};
        const char* out_names[] = {"mask", "state_out"};
        auto outputs = session_.Run(Ort::RunOptions{}, in_names, inputs, 2,
                                    out_names, 2);

        const float* m = outputs[0].GetTensorData<float>();
        std::copy(m, m + kBins, mask_.begin());
        const float* s = outputs[1].GetTensorData<float>();
        std::copy(s, s + state_.size(), state_.begin());
    }
    Ort::Env env_;
    Ort::Session session_;
    std::vector<float> window_, in_buf_, ola_buf_, norm_buf_;
    std::vector<dsp::cpx> frame_;
    std::vector<float> log_mag_;
    std::array<float, kBins> mask_{};
    std::vector<float> state_;
    std::vector<int64_t> state_dims_;
    Ort::MemoryInfo mem_info_;
};
}  // namespace rt
