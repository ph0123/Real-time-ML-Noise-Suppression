#include <chrono>
#include <iostream>
#include <vector>

#include "denoiser.hpp"
#include "dsp/wav.hpp"

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: offline_denoise model.onnx noisy.wav out.wav\n";
        return 1;
    }
    auto wav = dsp::read_wav(argv[2]);
    rt::Denoiser dn(argv[1]);

    const size_t hops = wav.samples.size() / rt::kHop;
    std::vector<float> out(hops * rt::kHop);

    auto t0 = std::chrono::steady_clock::now();
    for (size_t h = 0; h < hops; ++h)
        dn.process_hop(&wav.samples[h * rt::kHop], &out[h * rt::kHop]);
    auto t1 = std::chrono::steady_clock::now();

    const double total_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double audio_ms = hops * rt::kHop * 1000.0 / wav.sample_rate;
    std::cout << "Processed " << audio_ms / 1000 << " s of audio in "
              << total_ms / 1000 << " s\n"
              << "Per-hop: " << total_ms / hops << " ms (budget: 8 ms)\n"
              << "Real-time factor: " << total_ms / audio_ms << "\n";

    dsp::WavData res{wav.sample_rate, std::move(out)};
    dsp::write_wav(argv[3], res);
    return 0;
}
