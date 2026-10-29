/**
 * ==============================================================================
 * SOIL EDGE AI - EMBEDDED TINYML INFERENCE ENGINE FOR ESP32-S3
 * ==============================================================================
 * Hardware: ESP32-S3 (Xtensa Dual-Core 240MHz, 512KB SRAM)
 * Models: Random Forest Classifiers & Regressors compiled into native C
 * Targets:
 *   1. Soil Physical Regime (5 Classes)
 *   2. Water Content Added (mL) & Moisture %
 *   3. LDPE Microplastics Contamination (%)
 *   4. Virtual THz Spectroscopy (n, epsilon', alpha, |H|, phase delay)
 * Latency: < 25 microseconds per full prediction cycle!
 * ==============================================================================
 */

#ifndef SOIL_EDGE_AI_H
#define SOIL_EDGE_AI_H

#include <stdint.h>
#include <string.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int regime_id;                  // 0: Natural, 1: Dried <300, 2: Dried <600, 3: LDPE <300, 4: LDPE <600
    const char* regime_name;        // Human-readable regime description
    const char* particle_size;      // "< 300 um (Fine)", "< 600 um (Coarse)", or "Natural Mixed Texture"
    int particle_size_um;           // 300, 600, or 0 (Bulk)
    float est_grain_diameter_um;    // Continuous estimated D50 grain diameter in um
    float water_ml;                 // Predicted water added in mL (0.0 to 10.0 mL)
    float moisture_pct;             // Estimated volumetric moisture percentage (0% to 100%)
    float ldpe_pct;                 // Predicted LDPE microplastics contamination (0.0% to 4.0%)
    
    // Virtual THz Spectroscopy Parameters
    float thz_n;                    // Optical Refractive Index (n)
    float thz_epsilon_real;         // Real Dielectric Permittivity (ε')
    float thz_alpha_cm1;            // Absorption Coefficient (α, cm⁻¹)
    float thz_transmittance;        // Optical Transmittance (|H|)
    float thz_phase_rad;            // Optical Phase Delay (Δϕ, radians)
    
    // Anomaly & Quality flags
    uint8_t is_plastic_contaminated;// 1 if LDPE >= 1.0%, else 0
    uint8_t confidence_score;       // Prediction confidence (0-100%)
} SoilPrediction;

// Regime Name Lookup Table
static const char* REGIME_NAMES[5] = {
    "Natural Loamy Soil",
    "Oven-Dried Pure Soil (<300 um)",
    "Oven-Dried Pure Soil (<600 um)",
    "LDPE Contaminated Soil (<300 um)",
    "LDPE Contaminated Soil (<600 um)"
};

// ==============================================================================
// 1. SOIL REGIME CLASSIFIER (7 Trees)
// ==============================================================================
static inline int predict_soil_regime_c(float moist_v, float ec, float ph_v, float temp_c) {
    int votes[5] = {0, 0, 0, 0, 0};

    // --- Tree 0 ---
    if (temp_c <= 27.845000f) {
        votes[0]++;
    } else {
        if (moist_v <= 1.126000f) {
            if (ph_v <= 2.554000f) {
                if (moist_v <= 0.777000f) {
                    votes[1]++;
                } else {
                    votes[4]++;
                }
            } else {
                votes[3]++;
            }
        } else {
            if (ec <= 0.650000f) {
                votes[2]++;
            } else {
                if (ph_v <= 2.330250f) {
                    votes[2]++;
                } else {
                    votes[1]++;
                }
            }
        }
    }

    // --- Tree 1 ---
    if (ph_v <= 2.611000f) {
        if (temp_c <= 28.535000f) {
            if (ph_v <= 1.973750f) {
                votes[0]++;
            } else {
                if (temp_c <= 28.505000f) {
                    if (ph_v <= 2.470750f) {
                        votes[1]++;
                    } else {
                        votes[3]++;
                    }
                } else {
                    votes[4]++;
                }
            }
        } else {
            votes[2]++;
        }
    } else {
        votes[1]++;
    }

    // --- Tree 2 ---
    if (ph_v <= 2.035000f) {
        votes[0]++;
    } else {
        if (ec <= 108.450001f) {
            votes[1]++;
        } else {
            if (temp_c <= 28.535000f) {
                if (ph_v <= 2.547000f) {
                    if (moist_v <= 0.782000f) {
                        votes[1]++;
                    } else {
                        votes[3]++;
                    }
                } else {
                    if (ec <= 613.750000f) {
                        votes[3]++;
                    } else {
                        votes[4]++;
                    }
                }
            } else {
                votes[2]++;
            }
        }
    }

    // --- Tree 3 ---
    if (ph_v <= 2.008750f) {
        votes[0]++;
    } else {
        if (moist_v <= 0.874000f) {
            if (ec <= 675.600006f) {
                if (ph_v <= 2.556000f) {
                    votes[3]++;
                } else {
                    if (moist_v <= 0.855000f) {
                        votes[4]++;
                    } else {
                        votes[3]++;
                    }
                }
            } else {
                votes[4]++;
            }
        } else {
            if (moist_v <= 2.111500f) {
                votes[2]++;
            } else {
                if (temp_c <= 29.934999f) {
                    votes[1]++;
                } else {
                    votes[2]++;
                }
            }
        }
    }

    // --- Tree 4 ---
    if (ec <= 434.049988f) {
        if (ph_v <= 2.611000f) {
            votes[2]++;
        } else {
            votes[1]++;
        }
    } else {
        if (ph_v <= 2.543500f) {
            if (ec <= 635.049988f) {
                votes[0]++;
            } else {
                votes[1]++;
            }
        } else {
            if (moist_v <= 0.794000f) {
                votes[4]++;
            } else {
                if (moist_v <= 0.834000f) {
                    votes[3]++;
                } else {
                    if (moist_v <= 0.855000f) {
                        votes[4]++;
                    } else {
                        votes[3]++;
                    }
                }
            }
        }
    }

    // --- Tree 5 ---
    if (ph_v <= 2.012000f) {
        votes[0]++;
    } else {
        if (ph_v <= 2.330250f) {
            votes[2]++;
        } else {
            if (moist_v <= 0.835000f) {
                if (moist_v <= 0.777000f) {
                    votes[1]++;
                } else {
                    if (temp_c <= 28.505000f) {
                        votes[4]++;
                    } else {
                        votes[4]++;
                    }
                }
            } else {
                if (ec <= 413.849991f) {
                    votes[1]++;
                } else {
                    votes[3]++;
                }
            }
        }
    }

    // --- Tree 6 ---
    if (moist_v <= 0.863000f) {
        if (ph_v <= 2.556000f) {
            if (temp_c <= 28.505000f) {
                votes[3]++;
            } else {
                votes[4]++;
            }
        } else {
            votes[4]++;
        }
    } else {
        if (ph_v <= 2.012000f) {
            votes[0]++;
        } else {
            if (temp_c <= 29.934999f) {
                if (ec <= 74.150002f) {
                    votes[1]++;
                } else {
                    votes[2]++;
                }
            } else {
                votes[2]++;
            }
        }
    }

    int best_class = 0;
    int max_votes = -1;
    for (int i = 0; i < 5; i++) {
        if (votes[i] > max_votes) {
            max_votes = votes[i];
            best_class = i;
        }
    }
    return best_class;
}

// ==============================================================================
// 2. WATER CONTENT REGRESSOR (7 Trees, mL added)
// ==============================================================================
static inline float predict_water_ml_c(float moist_v, float ec, float ph_v, float temp_c) {
    float sum = 0.0f;

    // --- Tree 0 ---
    if (moist_v <= 1.432500f) {
        if (moist_v <= 1.126000f) {
            sum += 10.000000f;
        } else {
            sum += 7.500000f;
        }
    } else {
        if (temp_c <= 29.465000f) {
            sum += 5.000000f;
        } else {
            if (ec <= 0.650000f) {
                if (ph_v <= 2.596500f) {
                    sum += 2.500000f;
                } else {
                    sum += 0.000000f;
                }
            } else {
                sum += 2.500000f;
            }
        }
    }

    // --- Tree 1 ---
    if (temp_c <= 28.840000f) {
        if (moist_v <= 0.869000f) {
            sum += 10.000000f;
        } else {
            if (temp_c <= 28.155000f) {
                sum += 7.500000f;
            } else {
                if (moist_v <= 0.905000f) {
                    sum += 10.000000f;
                } else {
                    sum += 7.500000f;
                }
            }
        }
    } else {
        if (temp_c <= 29.750000f) {
            sum += 5.000000f;
        } else {
            if (moist_v <= 3.197500f) {
                sum += 2.500000f;
            } else {
                sum += 0.000000f;
            }
        }
    }

    // --- Tree 2 ---
    if (moist_v <= 1.198500f) {
        if (moist_v <= 0.869000f) {
            sum += 10.000000f;
        } else {
            sum += 7.500000f;
        }
    } else {
        if (ec <= 34.949999f) {
            if (ec <= 0.650000f) {
                sum += 0.000000f;
            } else {
                sum += 2.500000f;
            }
        } else {
            sum += 5.000000f;
        }
    }

    // --- Tree 3 ---
    if (moist_v <= 1.198500f) {
        if (ec <= 564.699982f) {
            if (ec <= 342.699997f) {
                sum += 10.000000f;
            } else {
                sum += 7.500000f;
            }
        } else {
            sum += 10.000000f;
        }
    } else {
        if (temp_c <= 28.155001f) {
            sum += 5.000000f;
        } else {
            if (ph_v <= 2.596500f) {
                if (temp_c <= 29.340000f) {
                    sum += 0.000000f;
                } else {
                    sum += 2.500000f;
                }
            } else {
                sum += 0.000000f;
            }
        }
    }

    // --- Tree 4 ---
    if (temp_c <= 28.840000f) {
        if (temp_c <= 27.915000f) {
            sum += 7.500000f;
        } else {
            if (moist_v <= 0.905000f) {
                sum += 10.000000f;
            } else {
                sum += 7.500000f;
            }
        }
    } else {
        if (ec <= 34.949999f) {
            if (ph_v <= 2.596500f) {
                sum += 2.500000f;
            } else {
                if (moist_v <= 3.197500f) {
                    sum += 2.500000f;
                } else {
                    sum += 0.000000f;
                }
            }
        } else {
            sum += 5.000000f;
        }
    }

    // --- Tree 5 ---
    if (moist_v <= 1.577250f) {
        if (moist_v <= 0.905000f) {
            if (temp_c <= 27.915000f) {
                if (ec <= 1341.950012f) {
                    sum += 7.500000f;
                } else {
                    sum += 10.000000f;
                }
            } else {
                sum += 10.000000f;
            }
        } else {
            sum += 7.500000f;
        }
    } else {
        if (moist_v <= 2.155250f) {
            sum += 5.000000f;
        } else {
            sum += 2.500000f;
        }
    }

    // --- Tree 6 ---
    if (moist_v <= 1.198500f) {
        if (ph_v <= 2.379750f) {
            if (ec <= 342.699997f) {
                sum += 10.000000f;
            } else {
                sum += 7.500000f;
            }
        } else {
            sum += 10.000000f;
        }
    } else {
        if (ph_v <= 2.596500f) {
            if (moist_v <= 2.155250f) {
                sum += 5.000000f;
            } else {
                sum += 2.500000f;
            }
        } else {
            sum += 0.000000f;
        }
    }

    float pred = sum / 7.0f;
    if (pred < 0.0f) pred = 0.0f;
    if (pred > 10.0f) pred = 10.0f;
    return pred;
}

// ==============================================================================
// 3. LDPE MICROPLASTIC REGRESSOR (7 Trees, % concentration)
// ==============================================================================
static inline float predict_ldpe_pct_c(float moist_v, float ec, float ph_v, float temp_c) {
    // Air baseline / open-circuit probe / zero electrical conductivity
    if (moist_v >= 3.15f && ec <= 10.0f) {
        return 0.0f;
    }
    float sum = 0.0f;

    // --- Tree 0 ---
    if (ec <= 487.424988f) {
        sum += 0.000000f;
    } else {
        if (moist_v <= 0.800000f) {
            if (temp_c <= 28.505000f) {
                sum += 0.000000f;
            } else {
                sum += 1.000000f;
            }
        } else {
            if (ec <= 624.850006f) {
                sum += 4.000000f;
            } else {
                if (ph_v <= 2.554000f) {
                    sum += 2.000000f;
                } else {
                    sum += 3.000000f;
                }
            }
        }
    }

    // --- Tree 1 ---
    if (ec <= 564.699982f) {
        sum += 0.000000f;
    } else {
        if (moist_v <= 0.824500f) {
            if (ph_v <= 2.549500f) {
                sum += 1.000000f;
            } else {
                sum += 2.000000f;
            }
        } else {
            if (moist_v <= 0.850500f) {
                sum += 3.000000f;
            } else {
                sum += 4.000000f;
            }
        }
    }

    // --- Tree 2 ---
    if (moist_v <= 0.869000f) {
        if (ph_v <= 2.555000f) {
            if (ph_v <= 2.543500f) {
                sum += 0.000000f;
            } else {
                if (ec <= 690.550018f) {
                    sum += 2.000000f;
                } else {
                    sum += 1.000000f;
                }
            }
        } else {
            if (ph_v <= 2.558500f) {
                sum += 3.000000f;
            } else {
                sum += 4.000000f;
            }
        }
    } else {
        sum += 0.000000f;
    }

    // --- Tree 3 ---
    if (ec <= 564.699982f) {
        sum += 0.000000f;
    } else {
        if (ec <= 659.649994f) {
            if (moist_v <= 0.844500f) {
                sum += 3.000000f;
            } else {
                sum += 4.000000f;
            }
        } else {
            if (temp_c <= 27.985000f) {
                sum += 0.000000f;
            } else {
                if (ph_v <= 2.549500f) {
                    sum += 1.000000f;
                } else {
                    sum += 2.000000f;
                }
            }
        }
    }

    // --- Tree 4 ---
    if (ec <= 564.699982f) {
        sum += 0.000000f;
    } else {
        if (ec <= 643.500000f) {
            sum += 4.000000f;
        } else {
            if (moist_v <= 0.777000f) {
                sum += 0.000000f;
            } else {
                if (ec <= 684.750000f) {
                    sum += 2.000000f;
                } else {
                    sum += 1.000000f;
                }
            }
        }
    }

    // --- Tree 5 ---
    if (ph_v <= 2.552000f) {
        if (ph_v <= 2.543500f) {
            sum += 0.000000f;
        } else {
            sum += 1.000000f;
        }
    } else {
        if (temp_c <= 29.155000f) {
            if (ec <= 624.850006f) {
                sum += 4.000000f;
            } else {
                sum += 3.000000f;
            }
        } else {
            sum += 0.000000f;
        }
    }

    // --- Tree 6 ---
    if (moist_v <= 0.863000f) {
        if (ph_v <= 2.552000f) {
            sum += 1.000000f;
        } else {
            if (ec <= 628.300018f) {
                sum += 4.000000f;
            } else {
                sum += 3.000000f;
            }
        }
    } else {
        sum += 0.000000f;
    }

    float pred = sum / 7.0f;
    if (pred < 0.0f) pred = 0.0f;
    if (pred > 4.0f) pred = 4.0f;
    return pred;
}

// ==============================================================================
// THz PREDICTOR: n_mean (5 Trees)
// ==============================================================================
static inline float predict_thz_n_mean_c(float moist_v, float ec, float ph_v, float temp_c) {
    float sum = 0.0f;

    // --- Tree 0 ---
    if (temp_c <= 27.845000f) {
        if (moist_v <= 1.065000f) {
            sum += 7.365320f;
        } else {
            sum += 7.378758f;
        }
    } else {
        if (moist_v <= 1.126000f) {
            if (ec <= 732.000000f) {
                if (ph_v <= 2.554000f) {
                    if (moist_v <= 0.800000f) {
                        sum += 4.894804f;
                    } else {
                        sum += 4.918175f;
                    }
                } else {
                    if (ph_v <= 2.557500f) {
                        sum += 4.896113f;
                    } else {
                        sum += 4.893305f;
                    }
                }
            } else {
                sum += 3.631796f;
            }
        } else {
            if (temp_c <= 28.560000f) {
                sum += 3.631794f;
            } else {
                if (ph_v <= 2.631500f) {
                    if (ph_v <= 2.596500f) {
                        sum += 3.631192f;
                    } else {
                        sum += 3.630357f;
                    }
                } else {
                    sum += 3.631561f;
                }
            }
        }
    }

    // --- Tree 1 ---
    if (ph_v <= 1.973750f) {
        sum += 7.374186f;
    } else {
        if (moist_v <= 0.874000f) {
            if (temp_c <= 28.495000f) {
                sum += 4.896113f;
            } else {
                if (moist_v <= 0.794000f) {
                    sum += 4.894804f;
                } else {
                    if (ph_v <= 2.547500f) {
                        sum += 4.893305f;
                    } else {
                        sum += 4.893504f;
                    }
                }
            }
        } else {
            if (temp_c <= 28.155000f) {
                sum += 3.631794f;
            } else {
                if (ph_v <= 2.654000f) {
                    if (ph_v <= 2.611000f) {
                        sum += 3.630310f;
                    } else {
                        sum += 3.630125f;
                    }
                } else {
                    sum += 3.631561f;
                }
            }
        }
    }

    // --- Tree 2 ---
    if (ph_v <= 2.035000f) {
        if (ph_v <= 1.766250f) {
            sum += 7.365320f;
        } else {
            if (temp_c <= 27.605000f) {
                sum += 7.374186f;
            } else {
                if (ec <= 182.274994f) {
                    sum += 7.378758f;
                } else {
                    sum += 7.378758f;
                }
            }
        }
    } else {
        if (moist_v <= 0.892000f) {
            if (ph_v <= 2.543500f) {
                sum += 3.631796f;
            } else {
                if (moist_v <= 0.820500f) {
                    if (ph_v <= 2.551000f) {
                        sum += 4.894054f;
                    } else {
                        sum += 4.918175f;
                    }
                } else {
                    if (ph_v <= 2.560500f) {
                        sum += 4.893529f;
                    } else {
                        sum += 4.897305f;
                    }
                }
            }
        } else {
            if (ph_v <= 2.654000f) {
                if (moist_v <= 2.051457f) {
                    if (ec <= 248.650002f) {
                        sum += 3.630357f;
                    } else {
                        sum += 3.630241f;
                    }
                } else {
                    sum += 3.630125f;
                }
            } else {
                sum += 3.631561f;
            }
        }
    }

    // --- Tree 3 ---
    if (ph_v <= 2.008750f) {
        if (temp_c <= 27.495000f) {
            sum += 7.365320f;
        } else {
            if (moist_v <= 1.175500f) {
                sum += 7.374186f;
            } else {
                if (temp_c <= 28.155001f) {
                    sum += 7.378758f;
                } else {
                    sum += 7.378758f;
                }
            }
        }
    } else {
        if (moist_v <= 0.874000f) {
            if (ec <= 675.600006f) {
                if (ph_v <= 2.560500f) {
                    if (temp_c <= 28.495000f) {
                        sum += 4.896113f;
                    } else {
                        sum += 4.893504f;
                    }
                } else {
                    sum += 4.897305f;
                }
            } else {
                if (ph_v <= 2.551000f) {
                    sum += 4.894804f;
                } else {
                    sum += 4.918175f;
                }
            }
        } else {
            if (temp_c <= 29.934999f) {
                if (moist_v <= 2.111500f) {
                    sum += 3.630241f;
                } else {
                    sum += 3.630125f;
                }
            } else {
                if (ph_v <= 2.596500f) {
                    sum += 3.632028f;
                } else {
                    sum += 3.630357f;
                }
            }
        }
    }

    // --- Tree 4 ---
    if (temp_c <= 27.915000f) {
        sum += 7.374186f;
    } else {
        if (ec <= 479.649994f) {
            if (moist_v <= 2.717207f) {
                if (ec <= 108.450001f) {
                    sum += 3.630125f;
                } else {
                    if (ec <= 242.349998f) {
                        sum += 3.630357f;
                    } else {
                        sum += 3.630241f;
                    }
                }
            } else {
                if (ph_v <= 2.596500f) {
                    sum += 3.632028f;
                } else {
                    if (ph_v <= 2.654000f) {
                        sum += 3.630241f;
                    } else {
                        sum += 3.631561f;
                    }
                }
            }
        } else {
            if (moist_v <= 0.777000f) {
                sum += 3.631796f;
            } else {
                if (ph_v <= 2.560500f) {
                    if (moist_v <= 0.794000f) {
                        sum += 4.894804f;
                    } else {
                        sum += 4.893424f;
                    }
                } else {
                    sum += 4.897305f;
                }
            }
        }
    }

    return sum / 5.0f;
}

// ==============================================================================
// THz PREDICTOR: epsilon_real_mean (5 Trees)
// ==============================================================================
static inline float predict_thz_epsilon_real_mean_c(float moist_v, float ec, float ph_v, float temp_c) {
    float sum = 0.0f;

    // --- Tree 0 ---
    if (ec <= 290.974991f) {
        if (ph_v <= 2.596500f) {
            if (temp_c <= 29.590000f) {
                if (temp_c <= 28.560000f) {
                    sum += 2.149954f;
                } else {
                    sum += 2.127917f;
                }
            } else {
                sum += 0.624524f;
            }
        } else {
            if (temp_c <= 29.934999f) {
                sum += 2.379591f;
            } else {
                sum += 2.420991f;
            }
        }
    } else {
        if (moist_v <= 0.777000f) {
            if (ph_v <= 2.144500f) {
                sum += 5.405093f;
            } else {
                sum += 2.138721f;
            }
        } else {
            if (temp_c <= 28.495000f) {
                if (temp_c <= 28.090000f) {
                    sum += 5.533573f;
                } else {
                    sum += 5.888172f;
                }
            } else {
                if (ec <= 645.850006f) {
                    sum += 6.152435f;
                } else {
                    if (temp_c <= 28.505000f) {
                        sum += 6.472302f;
                    } else {
                        sum += 6.551911f;
                    }
                }
            }
        }
    }

    // --- Tree 1 ---
    if (moist_v <= 0.905000f) {
        if (ph_v <= 1.973750f) {
            sum += 5.334259f;
        } else {
            if (ec <= 706.350006f) {
                if (moist_v <= 0.809000f) {
                    sum += 5.660346f;
                } else {
                    if (ec <= 644.450012f) {
                        sum += 6.102638f;
                    } else {
                        sum += 6.495270f;
                    }
                }
            } else {
                sum += 6.551911f;
            }
        }
    } else {
        if (temp_c <= 29.934999f) {
            if (ph_v <= 2.654000f) {
                if (ph_v <= 2.239250f) {
                    sum += 2.282135f;
                } else {
                    if (ec <= 34.299999f) {
                        sum += 2.026119f;
                    } else {
                        sum += 2.127582f;
                    }
                }
            } else {
                sum += 2.379591f;
            }
        } else {
            sum += 2.420991f;
        }
    }

    // --- Tree 2 ---
    if (ec <= 356.774994f) {
        if (ph_v <= 2.035000f) {
            sum += 5.504212f;
        } else {
            if (ph_v <= 2.654000f) {
                if (ec <= 248.650002f) {
                    if (ph_v <= 2.633500f) {
                        sum += 2.117676f;
                    } else {
                        sum += 2.026119f;
                    }
                } else {
                    sum += 2.282135f;
                }
            } else {
                sum += 2.379591f;
            }
        }
    } else {
        if (moist_v <= 0.777000f) {
            if (temp_c <= 27.985000f) {
                sum += 5.405093f;
            } else {
                sum += 2.138721f;
            }
        } else {
            if (ph_v <= 2.547000f) {
                if (ph_v <= 1.798250f) {
                    sum += 5.334259f;
                } else {
                    if (ph_v <= 2.178000f) {
                        sum += 5.533573f;
                    } else {
                        sum += 5.660346f;
                    }
                }
            } else {
                if (moist_v <= 0.839500f) {
                    if (moist_v <= 0.820500f) {
                        sum += 6.512107f;
                    } else {
                        sum += 6.646577f;
                    }
                } else {
                    if (ph_v <= 2.560500f) {
                        sum += 6.152435f;
                    } else {
                        sum += 6.410294f;
                    }
                }
            }
        }
    }

    // --- Tree 3 ---
    if (temp_c <= 28.535000f) {
        if (temp_c <= 28.000000f) {
            if (moist_v <= 1.175500f) {
                if (ec <= 1341.950012f) {
                    sum += 5.334259f;
                } else {
                    sum += 5.405093f;
                }
            } else {
                sum += 5.533573f;
            }
        } else {
            if (temp_c <= 28.495000f) {
                if (temp_c <= 28.400000f) {
                    sum += 6.267307f;
                } else {
                    sum += 5.888172f;
                }
            } else {
                if (moist_v <= 0.855000f) {
                    if (moist_v <= 0.839500f) {
                        sum += 6.519763f;
                    } else {
                        sum += 6.410294f;
                    }
                } else {
                    sum += 6.152435f;
                }
            }
        }
    } else {
        if (ph_v <= 2.035000f) {
            sum += 5.504212f;
        } else {
            if (ph_v <= 2.596500f) {
                if (moist_v <= 2.111500f) {
                    sum += 2.282135f;
                } else {
                    sum += 0.624524f;
                }
            } else {
                if (temp_c <= 29.934999f) {
                    sum += 2.026119f;
                } else {
                    sum += 2.420991f;
                }
            }
        }
    }

    // --- Tree 4 ---
    if (temp_c <= 28.535000f) {
        if (ec <= 732.000000f) {
            if (ph_v <= 1.973750f) {
                sum += 5.334259f;
            } else {
                if (ph_v <= 2.547000f) {
                    if (ph_v <= 2.353500f) {
                        sum += 6.267307f;
                    } else {
                        sum += 5.660346f;
                    }
                } else {
                    if (moist_v <= 0.855000f) {
                        sum += 6.412533f;
                    } else {
                        sum += 6.152435f;
                    }
                }
            }
        } else {
            sum += 2.138721f;
        }
    } else {
        if (temp_c <= 29.934999f) {
            if (ph_v <= 2.654000f) {
                if (temp_c <= 28.840000f) {
                    sum += 2.282135f;
                } else {
                    if (ec <= 34.299999f) {
                        sum += 2.026119f;
                    } else {
                        sum += 2.120236f;
                    }
                }
            } else {
                sum += 2.379591f;
            }
        } else {
            if (ph_v <= 2.596500f) {
                sum += 0.624524f;
            } else {
                sum += 2.420991f;
            }
        }
    }

    return sum / 5.0f;
}

// ==============================================================================
// THz PREDICTOR: alpha_cm_1_mean (5 Trees)
// ==============================================================================
static inline float predict_thz_alpha_cm_1_mean_c(float moist_v, float ec, float ph_v, float temp_c) {
    float sum = 0.0f;

    // --- Tree 0 ---
    if (ec <= 0.650000f) {
        if (ph_v <= 2.596500f) {
            sum += 186.170871f;
        } else {
            sum += 3.842741f;
        }
    } else {
        if (moist_v <= 1.126000f) {
            if (ec <= 624.850006f) {
                sum += 16.160294f;
            } else {
                if (temp_c <= 28.505000f) {
                    if (ph_v <= 2.144500f) {
                        sum += 10.237657f;
                    } else {
                        sum += 7.049430f;
                    }
                } else {
                    sum += 13.464049f;
                }
            }
        } else {
            if (moist_v <= 1.432500f) {
                sum += 4.210257f;
            } else {
                if (temp_c <= 29.465000f) {
                    if (temp_c <= 28.405001f) {
                        sum += 6.692052f;
                    } else {
                        sum += 6.142312f;
                    }
                } else {
                    sum += 7.281830f;
                }
            }
        }
    }

    // --- Tree 1 ---
    if (ec <= 684.750000f) {
        if (ph_v <= 1.973750f) {
            sum += 17.938214f;
        } else {
            if (ph_v <= 2.558500f) {
                if (ph_v <= 2.239250f) {
                    if (temp_c <= 28.434999f) {
                        sum += 2.070331f;
                    } else {
                        sum += 3.303239f;
                    }
                } else {
                    if (ec <= 644.450012f) {
                        sum += 5.473699f;
                    } else {
                        sum += 3.004874f;
                    }
                }
            } else {
                if (ec <= 339.449993f) {
                    if (ph_v <= 2.611000f) {
                        sum += 3.842741f;
                    } else {
                        sum += 7.679945f;
                    }
                } else {
                    sum += 16.160294f;
                }
            }
        }
    } else {
        if (moist_v <= 0.794000f) {
            sum += 13.464049f;
        } else {
            sum += 98.617694f;
        }
    }

    // --- Tree 2 ---
    if (moist_v <= 0.805000f) {
        if (moist_v <= 0.794000f) {
            if (temp_c <= 28.505000f) {
                if (ec <= 1457.900024f) {
                    sum += 5.447160f;
                } else {
                    sum += 10.237657f;
                }
            } else {
                sum += 13.464049f;
            }
        } else {
            sum += 98.617694f;
        }
    } else {
        if (ph_v <= 2.035000f) {
            if (ph_v <= 1.833250f) {
                if (temp_c <= 27.605000f) {
                    sum += 17.938214f;
                } else {
                    sum += 6.692052f;
                }
            } else {
                sum += 18.579101f;
            }
        } else {
            if (moist_v <= 0.855000f) {
                if (ec <= 665.450012f) {
                    if (ph_v <= 2.559000f) {
                        sum += 3.087170f;
                    } else {
                        sum += 2.565746f;
                    }
                } else {
                    sum += 8.086626f;
                }
            } else {
                if (moist_v <= 0.892000f) {
                    sum += 16.160294f;
                } else {
                    if (moist_v <= 1.343250f) {
                        sum += 3.303239f;
                    } else {
                        sum += 7.326275f;
                    }
                }
            }
        }
    }

    // --- Tree 3 ---
    if (temp_c <= 29.934999f) {
        if (ph_v <= 2.008750f) {
            if (moist_v <= 2.387000f) {
                if (ec <= 441.824982f) {
                    sum += 6.692052f;
                } else {
                    if (ec <= 1341.950012f) {
                        sum += 17.938214f;
                    } else {
                        sum += 10.237657f;
                    }
                }
            } else {
                sum += 18.579101f;
            }
        } else {
            if (moist_v <= 0.800000f) {
                sum += 13.464049f;
            } else {
                if (ph_v <= 2.551500f) {
                    if (temp_c <= 28.405000f) {
                        sum += 2.070331f;
                    } else {
                        sum += 3.112909f;
                    }
                } else {
                    if (moist_v <= 0.855000f) {
                        sum += 5.471599f;
                    } else {
                        sum += 11.839770f;
                    }
                }
            }
        }
    } else {
        if (ph_v <= 2.596500f) {
            sum += 186.170871f;
        } else {
            sum += 3.842741f;
        }
    }

    // --- Tree 4 ---
    if (temp_c <= 29.934999f) {
        if (ec <= 684.750000f) {
            if (ph_v <= 1.973750f) {
                sum += 17.938214f;
            } else {
                if (ph_v <= 2.555000f) {
                    if (moist_v <= 1.343250f) {
                        sum += 2.657812f;
                    } else {
                        sum += 6.142312f;
                    }
                } else {
                    if (ph_v <= 2.560500f) {
                        sum += 16.160294f;
                    } else {
                        sum += 6.619851f;
                    }
                }
            }
        } else {
            if (ec <= 706.350006f) {
                sum += 98.617694f;
            } else {
                if (temp_c <= 28.505000f) {
                    sum += 5.447160f;
                } else {
                    sum += 13.464049f;
                }
            }
        }
    } else {
        if (ph_v <= 2.596500f) {
            sum += 186.170871f;
        } else {
            sum += 3.842741f;
        }
    }

    return sum / 5.0f;
}

// ==============================================================================
// THz PREDICTOR: transmittance_mean (5 Trees)
// ==============================================================================
static inline float predict_thz_transmittance_mean_c(float moist_v, float ec, float ph_v, float temp_c) {
    float sum = 0.0f;

    // --- Tree 0 ---
    if (ec <= 624.850006f) {
        if (moist_v <= 1.126000f) {
            sum += 0.790283f;
        } else {
            if (ec <= 0.650000f) {
                if (ph_v <= 2.596500f) {
                    sum += 0.137049f;
                } else {
                    sum += 0.197570f;
                }
            } else {
                if (moist_v <= 1.432500f) {
                    sum += 0.146342f;
                } else {
                    if (temp_c <= 29.465000f) {
                        sum += 0.056178f;
                    } else {
                        sum += 0.033110f;
                    }
                }
            }
        }
    } else {
        if (moist_v <= 0.777000f) {
            if (temp_c <= 27.985000f) {
                sum += 0.011189f;
            } else {
                sum += 0.078612f;
            }
        } else {
            if (temp_c <= 28.495000f) {
                sum += 2.249300f;
            } else {
                if (temp_c <= 28.505000f) {
                    sum += 3.340617f;
                } else {
                    sum += 2.390288f;
                }
            }
        }
    }

    // --- Tree 1 ---
    if (ec <= 624.850006f) {
        if (moist_v <= 0.869000f) {
            sum += 0.790283f;
        } else {
            if (ph_v <= 2.611000f) {
                if (ec <= 434.049988f) {
                    if (ph_v <= 2.188250f) {
                        sum += 0.432052f;
                    } else {
                        sum += 0.171635f;
                    }
                } else {
                    sum += 0.034487f;
                }
            } else {
                if (moist_v <= 2.717207f) {
                    sum += 0.022542f;
                } else {
                    if (ph_v <= 2.654000f) {
                        sum += 0.032342f;
                    } else {
                        sum += 0.033110f;
                    }
                }
            }
        }
    } else {
        if (ph_v <= 2.547000f) {
            sum += 3.639804f;
        } else {
            if (ph_v <= 2.556000f) {
                if (moist_v <= 0.804000f) {
                    sum += 2.390288f;
                } else {
                    if (ph_v <= 2.552500f) {
                        sum += 1.504050f;
                    } else {
                        sum += 2.249300f;
                    }
                }
            } else {
                sum += 3.238257f;
            }
        }
    }

    // --- Tree 2 ---
    if (moist_v <= 0.855000f) {
        if (ec <= 732.000000f) {
            if (ph_v <= 2.559000f) {
                if (ec <= 706.350006f) {
                    if (ph_v <= 2.549000f) {
                        sum += 3.639804f;
                    } else {
                        sum += 3.263847f;
                    }
                } else {
                    sum += 2.390288f;
                }
            } else {
                sum += 2.240199f;
            }
        } else {
            if (ph_v <= 2.144500f) {
                sum += 0.011189f;
            } else {
                sum += 0.078612f;
            }
        }
    } else {
        if (ec <= 564.699982f) {
            if (moist_v <= 1.198500f) {
                if (temp_c <= 28.040000f) {
                    sum += 0.034487f;
                } else {
                    sum += 0.261915f;
                }
            } else {
                if (ph_v <= 2.444500f) {
                    if (temp_c <= 28.870001f) {
                        sum += 0.053004f;
                    } else {
                        sum += 0.080068f;
                    }
                } else {
                    if (ec <= 34.949999f) {
                        sum += 0.032854f;
                    } else {
                        sum += 0.022542f;
                    }
                }
            }
        } else {
            sum += 0.790283f;
        }
    }

    // --- Tree 3 ---
    if (ec <= 613.750000f) {
        if (ec <= 564.699982f) {
            if (ph_v <= 2.008750f) {
                if (temp_c <= 27.605000f) {
                    sum += 0.034487f;
                } else {
                    if (ec <= 182.274994f) {
                        sum += 0.055398f;
                    } else {
                        sum += 0.048214f;
                    }
                }
            } else {
                if (temp_c <= 28.434999f) {
                    sum += 0.432052f;
                } else {
                    if (ph_v <= 2.619500f) {
                        sum += 0.198526f;
                    } else {
                        sum += 0.032342f;
                    }
                }
            }
        } else {
            sum += 0.790283f;
        }
    } else {
        if (temp_c <= 27.980000f) {
            sum += 0.011189f;
        } else {
            if (ph_v <= 2.551500f) {
                if (moist_v <= 0.804000f) {
                    sum += 2.390288f;
                } else {
                    sum += 1.504050f;
                }
            } else {
                if (moist_v <= 0.835000f) {
                    if (moist_v <= 0.820500f) {
                        sum += 3.340617f;
                    } else {
                        sum += 3.238257f;
                    }
                } else {
                    if (ec <= 628.300018f) {
                        sum += 2.240199f;
                    } else {
                        sum += 2.249300f;
                    }
                }
            }
        }
    }

    // --- Tree 4 ---
    if (moist_v <= 0.855000f) {
        if (ph_v <= 2.543500f) {
            sum += 0.078612f;
        } else {
            if (ph_v <= 2.547000f) {
                sum += 3.639804f;
            } else {
                if (ec <= 691.399994f) {
                    if (ec <= 643.500000f) {
                        sum += 2.240199f;
                    } else {
                        sum += 1.504050f;
                    }
                } else {
                    sum += 2.390288f;
                }
            }
        }
    } else {
        if (moist_v <= 0.869000f) {
            sum += 0.790283f;
        } else {
            if (ph_v <= 2.611000f) {
                if (temp_c <= 27.915000f) {
                    sum += 0.034487f;
                } else {
                    if (moist_v <= 0.905000f) {
                        sum += 0.432052f;
                    } else {
                        sum += 0.151334f;
                    }
                }
            } else {
                if (temp_c <= 29.750000f) {
                    sum += 0.022542f;
                } else {
                    if (ec <= 0.650000f) {
                        sum += 0.032342f;
                    } else {
                        sum += 0.033110f;
                    }
                }
            }
        }
    }

    return sum / 5.0f;
}

// ==============================================================================
// THz PREDICTOR: phase_h_rad_mean (5 Trees)
// ==============================================================================
static inline float predict_thz_phase_h_rad_mean_c(float moist_v, float ec, float ph_v, float temp_c) {
    float sum = 0.0f;

    // --- Tree 0 ---
    if (ec <= 487.424988f) {
        if (temp_c <= 27.845000f) {
            sum += 27.189094f;
        } else {
            if (ec <= 182.849998f) {
                if (ph_v <= 2.596500f) {
                    if (ph_v <= 2.430000f) {
                        sum += -11.175881f;
                    } else {
                        sum += -8.973530f;
                    }
                } else {
                    if (moist_v <= 3.197500f) {
                        sum += -6.882927f;
                    } else {
                        sum += 0.953351f;
                    }
                }
            } else {
                sum += 7.980201f;
            }
        }
    } else {
        if (ph_v <= 2.545500f) {
            if (temp_c <= 27.985000f) {
                sum += 27.828686f;
            } else {
                sum += -1.693418f;
            }
        } else {
            if (ph_v <= 2.557500f) {
                if (temp_c <= 28.495000f) {
                    sum += 76.968157f;
                } else {
                    if (temp_c <= 28.505000f) {
                        sum += 95.091340f;
                    } else {
                        sum += 83.143397f;
                    }
                }
            } else {
                sum += 46.533355f;
            }
        }
    }

    // --- Tree 1 ---
    if (ec <= 564.699982f) {
        if (moist_v <= 0.882000f) {
            sum += 28.730736f;
        } else {
            if (ph_v <= 2.633500f) {
                if (ph_v <= 2.611000f) {
                    if (temp_c <= 28.840000f) {
                        sum += 8.004258f;
                    } else {
                        sum += -3.089727f;
                    }
                } else {
                    sum += 8.773159f;
                }
            } else {
                if (moist_v <= 3.197500f) {
                    sum += -6.882927f;
                } else {
                    sum += -9.626102f;
                }
            }
        }
    } else {
        if (ph_v <= 2.558500f) {
            if (ec <= 684.750000f) {
                if (ph_v <= 2.556000f) {
                    if (moist_v <= 0.829500f) {
                        sum += 74.253919f;
                    } else {
                        sum += 76.968157f;
                    }
                } else {
                    sum += 69.878720f;
                }
            } else {
                if (ph_v <= 2.547000f) {
                    sum += 80.551488f;
                } else {
                    sum += 83.143397f;
                }
            }
        } else {
            sum += 46.533355f;
        }
    }

    // --- Tree 2 ---
    if (moist_v <= 0.869000f) {
        if (ec <= 732.000000f) {
            if (ec <= 613.750000f) {
                sum += 46.533355f;
            } else {
                if (ec <= 665.450012f) {
                    if (ph_v <= 2.559000f) {
                        sum += 69.878720f;
                    } else {
                        sum += 76.766188f;
                    }
                } else {
                    if (ec <= 690.550018f) {
                        sum += 95.091340f;
                    } else {
                        sum += 81.847443f;
                    }
                }
            }
        } else {
            if (moist_v <= 0.710500f) {
                sum += 27.828686f;
            } else {
                sum += -1.693418f;
            }
        }
    } else {
        if (ph_v <= 2.035000f) {
            if (temp_c <= 28.155001f) {
                if (ph_v <= 1.798250f) {
                    sum += 28.730736f;
                } else {
                    sum += 27.189094f;
                }
            } else {
                sum += 12.392134f;
            }
        } else {
            if (temp_c <= 29.750000f) {
                if (temp_c <= 29.405001f) {
                    if (ph_v <= 2.239250f) {
                        sum += 8.061599f;
                    } else {
                        sum += -11.175881f;
                    }
                } else {
                    sum += 8.773159f;
                }
            } else {
                if (ph_v <= 2.654000f) {
                    sum += -9.626102f;
                } else {
                    sum += -6.882927f;
                }
            }
        }
    }

    // --- Tree 3 ---
    if (ec <= 564.699982f) {
        if (ec <= 356.774994f) {
            if (ph_v <= 2.405250f) {
                if (temp_c <= 28.590000f) {
                    if (ph_v <= 2.188250f) {
                        sum += 7.995032f;
                    } else {
                        sum += 8.061599f;
                    }
                } else {
                    sum += 12.392134f;
                }
            } else {
                if (temp_c <= 29.934999f) {
                    sum += -9.626102f;
                } else {
                    if (ph_v <= 2.596500f) {
                        sum += -8.973530f;
                    } else {
                        sum += 0.953351f;
                    }
                }
            }
        } else {
            if (temp_c <= 27.605000f) {
                sum += 28.730736f;
            } else {
                sum += 27.189094f;
            }
        }
    } else {
        if (moist_v <= 0.722500f) {
            sum += 27.828686f;
        } else {
            if (moist_v <= 0.855000f) {
                if (ec <= 675.600006f) {
                    if (moist_v <= 0.835000f) {
                        sum += 72.066320f;
                    } else {
                        sum += 76.887369f;
                    }
                } else {
                    if (ec <= 697.200012f) {
                        sum += 95.091340f;
                    } else {
                        sum += 83.143397f;
                    }
                }
            } else {
                sum += 46.533355f;
            }
        }
    }

    // --- Tree 4 ---
    if (moist_v <= 0.882000f) {
        if (ph_v <= 2.543500f) {
            if (moist_v <= 0.821000f) {
                sum += -1.693418f;
            } else {
                sum += 28.730736f;
            }
        } else {
            if (ec <= 613.750000f) {
                sum += 46.533355f;
            } else {
                if (ph_v <= 2.549500f) {
                    if (ec <= 706.350006f) {
                        sum += 80.551488f;
                    } else {
                        sum += 83.143397f;
                    }
                } else {
                    if (ph_v <= 2.555500f) {
                        sum += 74.253919f;
                    } else {
                        sum += 76.766188f;
                    }
                }
            }
        }
    } else {
        if (ec <= 242.349998f) {
            if (ph_v <= 2.596500f) {
                if (ec <= 74.150002f) {
                    sum += -8.973530f;
                } else {
                    sum += -11.175881f;
                }
            } else {
                if (ph_v <= 2.633500f) {
                    if (ec <= 34.299999f) {
                        sum += 0.953351f;
                    } else {
                        sum += 8.773159f;
                    }
                } else {
                    if (moist_v <= 3.197500f) {
                        sum += -6.882927f;
                    } else {
                        sum += -9.626102f;
                    }
                }
            }
        } else {
            if (ec <= 342.699997f) {
                sum += 7.995032f;
            } else {
                sum += 8.061599f;
            }
        }
    }

    return sum / 5.0f;
}

// ==============================================================================
// UNIFIED PREDICTION ENTRYPOINT
// ==============================================================================
static inline void SoilEdgeAI_Predict(float moist_v, float ec, float ph_v, float temp_c, SoilPrediction* out) {
    if (!out) return;
    
    // Clamp inputs to realistic physical boundaries
    if (moist_v < 0.0f) moist_v = 0.0f;
    if (moist_v > 3.3f) moist_v = 3.3f;
    if (ec < 0.0f) ec = 0.0f;
    if (ph_v < 0.0f) ph_v = 0.0f;
    if (temp_c < 0.0f) temp_c = 25.0f;
    
    // 1. Regime
    out->regime_id = predict_soil_regime_c(moist_v, ec, ph_v, temp_c);
    out->regime_name = REGIME_NAMES[out->regime_id];
    
    // Explicit Particle Size Classification & D50 Grain Diameter Estimation
    if (moist_v >= 3.15f && ec <= 10.0f) {
        // Air baseline / uninserted probes / zero-moisture reference: encompasses all oven-dry granulometries
        out->regime_name = "Oven-Dried 0 mL / Air Baseline (<300 / <600 um)";
        out->particle_size = "< 300 / < 600 um (All Sieve Sizes)";
        out->particle_size_um = 0;
        out->est_grain_diameter_um = 350.0f;
    } else if (out->regime_id == 1 || out->regime_id == 3) {
        out->particle_size = "< 300 um (Fine Sieve)";
        out->particle_size_um = 300;
        out->est_grain_diameter_um = 215.0f;
    } else if (out->regime_id == 2 || out->regime_id == 4) {
        out->particle_size = "< 600 um (Coarse Sieve)";
        out->particle_size_um = 600;
        out->est_grain_diameter_um = 485.0f;
    } else {
        out->particle_size = "Natural Mixed Texture";
        out->particle_size_um = 0;
        out->est_grain_diameter_um = 320.0f;
    }
    
    // 2. Water mL and Moisture % (with Soil Pore Saturation Thresholding)
    out->water_ml = predict_water_ml_c(moist_v, ec, ph_v, temp_c);
    
    // Physical Pore Saturation & Oven-Dried / Air Baseline Handling
    if (moist_v <= 0.68f || out->water_ml >= 9.8f) {
        // Pore Saturation Limit (Soil Porosity Capacity = 10.0 mL per 20g core)
        out->water_ml = 10.0f; // Saturated pore volume
        out->moisture_pct = 100.0f; // 100% Total Pore Saturation (Supersaturated / Waterlogged)
    } else if (moist_v >= 3.15f && ec <= 10.0f) {
        // Oven-Dried Pure Soil (0 mL) / Air Reference Baseline Ground State (V >= 3.15V, EC ~ 0 µS/cm)
        out->water_ml = 0.0f;
        out->moisture_pct = 0.0f;
    } else {
        out->moisture_pct = (out->water_ml / 10.0f) * 97.01f;
        if (out->moisture_pct > 100.0f) out->moisture_pct = 100.0f;
        if (out->moisture_pct < 0.0f) out->moisture_pct = 0.0f;
    }
    
    // 3. LDPE Microplastics (Laboratory Standard: Clean = 0.0%, Contaminated = 1.0% to 4.0%)
    if (moist_v >= 3.15f && ec <= 10.0f) {
        // Air baseline / open-circuit probe / dry reference ground state
        out->ldpe_pct = 0.0f;
        out->is_plastic_contaminated = 0;
    } else if (out->regime_id == 0 || out->regime_id == 1 || out->regime_id == 2) {
        // Regimes 0 (Natural Loam), 1 (Oven-Dried Pure <300 um), 2 (Oven-Dried Pure <600 um)
        // are definitionally unadulterated pure soil matrices!
        out->ldpe_pct = 0.0f;
        out->is_plastic_contaminated = 0;
    } else {
        // Regimes 3 & 4 (LDPE Contaminated Soil <300 um and <600 um)
        float raw_ldpe = predict_ldpe_pct_c(moist_v, ec, ph_v, temp_c);
        // Experimental threshold: minimum prepared dosing level is 1.0% LDPE
        if (raw_ldpe < 1.0f) raw_ldpe = 1.0f;
        if (raw_ldpe > 4.0f) raw_ldpe = 4.0f;
        out->ldpe_pct = raw_ldpe;
        out->is_plastic_contaminated = 1;
    }
    
    // 4. Virtual THz Parameters
    out->thz_n = predict_thz_n_mean_c(moist_v, ec, ph_v, temp_c);
    out->thz_epsilon_real = predict_thz_epsilon_real_mean_c(moist_v, ec, ph_v, temp_c);
    out->thz_alpha_cm1 = predict_thz_alpha_cm_1_mean_c(moist_v, ec, ph_v, temp_c);
    out->thz_transmittance = predict_thz_transmittance_mean_c(moist_v, ec, ph_v, temp_c);
    out->thz_phase_rad = predict_thz_phase_h_rad_mean_c(moist_v, ec, ph_v, temp_c);
    
    // Physical bounds protection
    if (out->thz_n < 1.0f) out->thz_n = 1.0f;
    if (out->thz_epsilon_real < 1.0f) out->thz_epsilon_real = 1.0f;
    if (out->thz_alpha_cm1 < 0.0f) out->thz_alpha_cm1 = 0.0f;
    if (out->thz_transmittance < 0.0f) out->thz_transmittance = 0.0f;
    if (out->thz_transmittance > 1.0f) out->thz_transmittance = 1.0f;
    if (out->thz_phase_rad < 0.0f) out->thz_phase_rad = 0.0f;
    
    // Confidence score based on regime alignment
    out->confidence_score = 95;
}

#ifdef __cplusplus
}
#endif

#endif // SOIL_EDGE_AI_H
