struct test_empty {
} e;
int Esize = sizeof(e);

struct rtx_def {
  unsigned short code;
  long long :3;
  int mode : 8;
  long long :0;
  long long x :31;
  //long long y:31;
} N = {2, 7, 1 };
int Nsize = sizeof(N);  // Size = 8

struct test1 {
  char x:1;
  long long :0;
} F1;  int F1size = sizeof(F1);  // Size = 4

struct test2 {
  long long x :4;
} F2;  int F2size = sizeof(F2);  // Size = 4

struct test3 {
  char x:1;
  long long :20;
} F3;  int F3size = sizeof(F3);  // Size = 3

struct test4 {
  char x:1;
  long long :21;
  short Y : 14;
} F4; int F4size = sizeof(F4);  // Size = 6

struct test5 {
  char x:1;
  long long :17;
  char Y : 1;
} F5; int F5size = sizeof(F5); // Size = 3

struct test6 {
  char x:1;
  long long :42;
  int Y : 21;
} F6; int F6size = sizeof(F6);  // Size = 8

struct test {
  char c;
  char d : 3;
  char e: 3;
  int : 0;
  char f;
  char :0;
  long long x : 4;
} M;   int Msize = sizeof(M);  // Size = 8

int main() {
  return 0;
}
