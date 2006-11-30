// This is a regression test on debug info to make sure that we can access 
// qualified global names.
// RUN: %llvmgcc -S -O0 -g %s -o - | llvm-as | llc --disable-fp-elim -o Output/Pubnames.s -f
// RUN: as Output/Pubnames.s -o Output/Pubnames.o
// RUN: g++ Output/Pubnames.o -o Output/Pubnames.exe
// RUN: ( echo "break main"; echo "run" ; echo "p Pubnames::pubname" ) > Output/Pubnames.gdbin 
// RUN: gdb -q -batch -n -x Output/Pubnames.gdbin Output/Pubnames.exe | tee Output/Pubnames.out | grep '10'
// XFAIL: i[1-9]86|alpha|ia64|arm

struct Pubnames {
  static int pubname;
};

int Pubnames::pubname = 10;

int main (int argc, char** argv) {
  Pubnames p;
  return 0;
}
