def check(output):
    C50 = 0
    SNR = 0
    total = 0
    for frame, (vad, snr, c50) in output:
        C50 += c50*(frame.end-frame.start)
        SNR += snr*(frame.end-frame.start)
        total += (frame.end-frame.start)

    C50 = C50/total
    SNR = SNR/total
    return C50, SNR
    
