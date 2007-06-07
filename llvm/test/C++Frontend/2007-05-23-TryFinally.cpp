// RUN: %llvmgxx %s -S -emit-llvm -O2 -o - | grep -c {handle\\|_Unwind_Resume} | grep {\[14\]}

struct One { };
struct Two { };

void handle_unexpected () {
  try
  {
    throw;
  }
  catch (One &)
  {
    throw Two ();
  }
}
