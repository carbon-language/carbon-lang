struct Y {
  int m;
  double f;
} y2;
enum E { e2 };

int g() {
  return y2.m + e2 + y2.f;
}
