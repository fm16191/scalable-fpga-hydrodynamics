#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cmath>

using std::string;
using std::stringstream;
using std::vector;

struct it_timers_t {
    double host_to_device = 0.0, boundaries_x = 0.0, boundaries_y = 0.0, device_to_host = 0.0, total_compute = 0.0,
           total_usage = 0.0, total_compute2;
};

struct stats_t { double sum{}, min{}, max{}, mean{}, stddev{}; };

// Timer stats calculation
static stats_t compute_stats(const vector<double>& values) {
    constexpr double zero_threshold = 1e-10;
    vector<double> filtered;
    filtered.reserve(values.size());
    for (double v : values)
        if (std::abs(v) > zero_threshold) filtered.push_back(v);

    if (filtered.empty()) return {};
    auto const [min_it, max_it] = std::minmax_element(filtered.begin(), filtered.end());
    const double sum  = std::accumulate(filtered.begin(), filtered.end(), 0.0);
    const double mean = sum / static_cast<double>(filtered.size());

    double sdev = 0.0;
    for (double v : filtered) sdev += (v - mean) * (v - mean);
    sdev = sqrt(sdev / static_cast<double>(filtered.size()));
    return {sum, *min_it, *max_it, mean, sdev};
}

// Print timer statistics
template<typename T>
static stats_t print_stats(const vector<it_timers_t>& timers, T it_timers_t::*member, const std::string& label) {
    vector<double> values;
    values.reserve(timers.size());
    for (const it_timers_t& t : timers)
        values.push_back(t.*member);

    stats_t stats = compute_stats(values);
    std::ostringstream buffer;

    buffer << "-----------------------------------------------------------\n"
        << label << "\n"
        << "Average execution time: (mean ± σ)   " << std::fixed << std::setprecision(1)
        << stats.mean << " us ± " << stats.stddev << " µs\n"
        << "                        (min … max)  " << stats.min << " us … " << stats.max << " µs\n\n";

    std::cout << buffer.str();

    return stats;
}

static double throughput(double bytes, double us) {
    if (us == 0.0) return 0.0;
    return bytes / (us / 1e6) / 1e9;
}
static double performance(double points, double us) {
    if (us == 0.0) return 0.0;
    return (points / (us / 1e6)) / 1e6;
}
static double ii(double us, double freq, double ops) {
    return (us / 1e6 * freq) / ops;
}

static void read_frequency(const char* exec_name, double *frequency)
{
    std::ifstream infile{ exec_name };
    if (!infile) {
        std::cerr << "Can't open design executable for frequency probe\n";
    }
    else {
        constexpr char tag[] = "Actual clock freq:";
        const auto tag_len = std::strlen(tag);
        std::string line;
        while (std::getline(infile, line)) {
            auto pos = line.find(tag);
            if (pos == std::string::npos) continue;

            double f = 0;
            // Scan first floating point after tag; leading spaces OK
            if (std::sscanf(line.c_str() + pos + tag_len, " %lf", &f) == 1 && f > 0.0) {
                *frequency = f * 1e6;
                printf("\nDesign frequency: %.2f MHz\n", *frequency / 1e6);
                break;
            }
        }
    }
}
