#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <vector>
#include <filesystem>

#include <romulus/common.h>

namespace stats {

struct result_t {
  // Latency
  double lat_min_us;
  double lat_max_us;
  double lat_avg_us;
  double lat_stddev_us;
  double lat_p50_us;
  double lat_p90_us;
  double lat_p95_us;
  double lat_p99_us;
  double lat_p99_9_us;
  uint64_t lat_min_idx;
  uint64_t lat_max_idx;
  // Throughput
  double thru_ops_per_s;
  // Bandwidth
  double band_mb_per_s;
  // Printing toggles
  bool report_lat;
  bool report_thru;
  bool report_band;
  std::string op_type;
  result_t()
      : lat_min_us(0.0),
        lat_max_us(0.0),
        lat_avg_us(0.0),
        lat_stddev_us(0.0),
        lat_p50_us(0.0),
        lat_p90_us(0.0),
        lat_p95_us(0.0),
        lat_p99_us(0.0),
        lat_p99_9_us(0.0),
        lat_min_idx(0),
        lat_max_idx(0),
        thru_ops_per_s(0.0),
        band_mb_per_s(0.0),
        report_lat(false),
        report_thru(false),
        report_band(false),
        op_type("") {}
  result_t(bool lat, bool thru, bool band) : result_t() {
    report_lat = lat;
    report_thru = thru;
    report_band = band;
  }

  std::string ToString() const {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(6);  // or higher
    if (report_lat) {
      ss << "\nLatency (us): min=" << lat_min_us << " (idx=" << lat_min_idx
         << "), max=" << lat_max_us << " (idx=" << lat_max_idx
         << "), avg=" << lat_avg_us << ", stddev=" << lat_stddev_us
         << ", p50=" << lat_p50_us << ", p90=" << lat_p90_us
         << ", p95=" << lat_p95_us << ", p99=" << lat_p99_us
         << ", p99.9=" << lat_p99_9_us;
    }
    if (report_thru) ss << "\nThroughput: " << thru_ops_per_s << " ops/s";

    if (report_band) ss << "\nBandwidth: " << band_mb_per_s << " MB/s";

    return ss.str();
  }
  void log_csv(const std::string &filename) const {
    // Check if file exists
    const bool file_exists = std::filesystem::exists(filename);
    std::ofstream ofs(filename, std::ios_base::app);
    ofs << std::fixed << std::setprecision(6);
    if (!file_exists) {
      // Write csv header
      ofs << "op_type,lat_min_us,lat_max_us,lat_avg_us,lat_stddev_us,"
             "lat_p50_us,lat_p90_us,lat_p95_us,lat_p99_us,lat_p99_9_us,"
             "thru_ops_per_s,band_mb_per_s\n";
    }
    ofs << op_type << "," << lat_min_us << "," << lat_max_us << "," << lat_avg_us
        << "," << lat_stddev_us << "," << lat_p50_us << "," << lat_p90_us
        << "," << lat_p95_us << "," << lat_p99_us << "," << lat_p99_9_us << ","
        << thru_ops_per_s << "," << band_mb_per_s << "\n";
  }
};

struct collector_t {
  std::string op_type;
  double total_time_s;
  std::vector<double> times;  // in us
  uint64_t ops;
  uint64_t bytes;
  collector_t() : op_type(""), total_time_s(0.0), ops(0), bytes(0) {}
};

inline void digest_latency(collector_t *data, result_t *result) {
  const auto &v = data->times;
  const size_t n = v.size();

  auto min_it = std::min_element(v.begin(), v.end());
  auto max_it = std::max_element(v.begin(), v.end());
  result->lat_min_us = *min_it;
  result->lat_max_us = *max_it;
  result->lat_min_idx = std::distance(v.begin(), min_it);
  result->lat_max_idx = std::distance(v.begin(), max_it);

  std::vector<double> sorted = v;
  std::sort(sorted.begin(), sorted.end());

  result->lat_avg_us = std::accumulate(sorted.begin(), sorted.end(), 0.0) / n;
  double sq_sum =
      std::inner_product(sorted.begin(), sorted.end(), sorted.begin(), 0.0);
  result->lat_stddev_us =
      std::sqrt(sq_sum / n - result->lat_avg_us * result->lat_avg_us);

  auto pct = [&](double p) { return sorted[(size_t)(p * (n - 1))]; };
  result->lat_p50_us = pct(0.50);
  result->lat_p90_us = pct(0.90);
  result->lat_p95_us = pct(0.95);
  result->lat_p99_us = pct(0.99);
  result->lat_p99_9_us = pct(0.999);
}

inline void digest_thru(collector_t *data, result_t *result) {
  result->thru_ops_per_s = (double)data->ops / data->total_time_s;
}

inline void digest_bandwidth(collector_t *data, result_t *result) {
  result->band_mb_per_s = (double)(data->bytes * 1e-6) / data->total_time_s;
}

inline result_t digest(collector_t *data, bool latency = true,
                       bool throughput = true, bool bandwidth = true) {
  ROMULUS_ASSERT(
      latency || throughput || bandwidth,
      "At least one of latency, throughput, or bandwidth must be true");
  result_t result(latency, throughput, bandwidth);
  result.op_type = data->op_type;
  if (latency) {
    ROMULUS_ASSERT(!data->times.empty(), "Invalid data for latency digest");
    digest_latency(data, &result);
  }
  if (throughput) {
    ROMULUS_ASSERT(data->ops > 0 && data->total_time_s > 0.0,
                   "Invalid data for throughput digest");
    digest_thru(data, &result);
  }
  if (bandwidth) {
    ROMULUS_ASSERT(data->bytes > 0 && data->total_time_s > 0.0,
                   "Invalid data for bandwidth digest");
    digest_bandwidth(data, &result);
  }
  return result;
}

}  // namespace stats