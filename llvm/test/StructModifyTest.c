typedef struct {
  int w;
  float x;
  double y;
  long long z;
} S1Ty;

typedef struct {
  S1Ty A, B;
} S2Ty;

void printS1(S1Ty *V) {
  printf("%d, %f, %f, %lld\n", V->w, V->x, V->y, V->z);
}

void main() {
  S2Ty E;
  E.A.w = 1;
  E.A.x = 123.42f;
  E.A.y = 19.0;
  E.A.z = 123455678902ll;
  E.B.w = 2;
  E.B.x = 23.42f;
  E.B.y = 29.0;
  E.B.z = 23455678902ll;

  printS1(&E.A);
  printS1(&E.B);
}

