// Minimal C++ check for SHtech Planner.hpp:
//     const int armor_jump_interval = 0.1f;
//
// Build/run:
//   g++ -O2 -std=c++17 -Wall verify/armor_jump_check.cpp -o /tmp/armor_jump_check
//   /tmp/armor_jump_check
//
// The program prints KEY=VALUE lines so verify_projects.py can parse them.

#include <iomanip>
#include <iostream>

int main()
{
    const int armor_jump_interval = 0.1f;  // same declaration as Planner.hpp:80
    const double dt_zero = 0.0;

    std::cout << "sizeof_int=" << sizeof(int) << "\n";
    std::cout << std::setprecision(17);
    std::cout << "float_0.1f_as_double=" << static_cast<double>(0.1f) << "\n";
    std::cout << "armor_jump_interval_as_int=" << armor_jump_interval << "\n";

    // armor_jump = !(dt_since_jump > armor_jump_interval).
    // Only dt == 0 can make the predicate false when the interval is 0.
    const bool dt_zero_gt_interval = dt_zero > armor_jump_interval;
    std::cout << "dt_zero_gt_interval=" << (dt_zero_gt_interval ? 1 : 0) << "\n";
    std::cout << "armor_jump_at_dt_zero=" << (dt_zero_gt_interval ? 0 : 1) << "\n";

    // In real code the second clock read happens after the first one, so the
    // jump frame already has dt > 0.  Simulate 200 frames at 1 kHz after a jump,
    // with the first evaluation 1 microsecond after armor_jump_tp was set.
    int post_jump_frames = 0;
    int armor_jump_true_frames = 0;
    for (int i = 0; i < 200; ++i) {
        const double dt = 1e-6 + i / 1000.0;
        ++post_jump_frames;
        const bool predicate = dt > armor_jump_interval;  // true => armor_jump = false
        if (!predicate) ++armor_jump_true_frames;
    }
    std::cout << "post_jump_frames=" << post_jump_frames << "\n";
    std::cout << "armor_jump_true_frames=" << armor_jump_true_frames << "\n";

    // Contrast: a correct double 0.1 s gate would keep armor_jump true for
    // roughly 100 frames at 1 kHz.
    int intended_true_frames = 0;
    for (int i = 0; i < 200; ++i) {
        const double dt = 1e-6 + i / 1000.0;
        if (!(dt > 0.1)) ++intended_true_frames;
    }
    std::cout << "intended_true_frames=" << intended_true_frames << "\n";
    return 0;
}
