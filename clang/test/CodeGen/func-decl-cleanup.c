// RUN: clang %s -emit-llvm -o -


// PR2360
typedef void fn_t();

fn_t a,b;

void b()
{
}

