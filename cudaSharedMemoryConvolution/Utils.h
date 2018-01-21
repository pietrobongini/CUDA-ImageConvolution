//
// Created by pietro bongini on 28/09/17.
//

#ifndef KERNELPROCESSING_UTILS_H
#define KERNELPROCESSING_UTILS_H

template<typename T>
static inline T _abs(const T &a) {
    return a < 0 ? -a : a;
}

static inline bool almostEqualFloat(float A, float B, float eps) {
    if (A == 0) {
        return _abs(B) < eps;
    } else if (B == 0) {
        return _abs(A) < eps;
    } else {
#if 0
        float d = max(_abs(A), _abs(B));
		float g = (_abs(A - B) / d);
#else
        float g = _abs(A - B);
#endif
        if (g <= eps) {
            return true;
        } else {
            return false;
        }
    }
}

static inline bool almostEqualFloat(float A, float B) {
    return almostEqualFloat(A, B, 0.2f);
}

static inline bool almostUnequalFloat(float a, float b) {
    return !almostEqualFloat(a, b);
}

static inline float _min(float x, float y) {
    return x < y ? x : y;
}

static inline float _max(float x, float y) {
    return x > y ? x : y;
}

static inline float clamp(float x, float start, float end) {
    return _min(_max(x, start), end);
}


#endif //KERNELPROCESSING_UTILS_H
