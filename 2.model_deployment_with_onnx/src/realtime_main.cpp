#include <portaudio.h>

#include <atomic>
#include <csignal>
#include <iostream>

#include "denoiser.hpp"
#include "dsp/ring_buffer.hpp"

// Audio callback only moves samples through lock-free ring buffers.
// All DSP + inference happens on a worker thread.
// This is THE real-time audio pattern: callback must never block.

static dsp::RingBuffer g_in(16384), g_out(16384);
static std::atomic<bool> g_run{true};

static int pa_callback(const void* input, void* output, unsigned long frames,
                       const PaStreamCallbackTimeInfo*, PaStreamCallbackFlags,
                       void*) {
    const float* in = static_cast<const float*>(input);
    float* out = static_cast<float*>(output);
    if (in) g_in.push(in, frames);
    if (!g_out.pop(out, frames))
        std::fill(out, out + frames, 0.0f);  // underrun -> silence
    return paContinue;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "Usage: realtime_denoise model.onnx\n"; return 1; }
    rt::Denoiser dn(argv[1]);

    Pa_Initialize();
    PaStream* stream;
    Pa_OpenDefaultStream(&stream, 1, 1, paFloat32, 16000,
                         rt::kHop, pa_callback, nullptr);
    Pa_StartStream(stream);
    std::signal(SIGINT, [](int) { g_run = false; });
    std::cout << "Running (Ctrl+C to stop). Speak with background noise...\n";

    float in_hop[rt::kHop], out_hop[rt::kHop];
    while (g_run) {
        if (g_in.readable() >= rt::kHop) {
            g_in.pop(in_hop, rt::kHop);
            dn.process_hop(in_hop, out_hop);
            g_out.push(out_hop, rt::kHop);
        } else {
            Pa_Sleep(1);
        }
    }

    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();
    return 0;
}
