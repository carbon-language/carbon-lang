/* RUN: clang-cc %s -ast-print
 */

void foo() {
  int X;
  X = sizeof(void (*(*)())());
  X = sizeof(int(*)(int, float, ...));
  X = sizeof(void (*(int arga, void (*argb)(double Y)))(void* Z));
}

