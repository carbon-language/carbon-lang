// RUN: %clang_cc1 -analyze -analyzer-checker=core -std=c++11 -verify %s

// radar://11485149, PR12871
class PlotPoint {
  bool valid;
};

PlotPoint limitedFit () {
  PlotPoint fit0;
  fit0 = limitedFit ();
  return fit0;
}
