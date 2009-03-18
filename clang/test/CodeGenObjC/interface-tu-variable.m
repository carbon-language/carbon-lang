// RUN: clang -fnext-runtime -emit-llvm -o %t %s
// RUN: grep 'two = global' %t &&
// RUN: grep 'ddd = common' %t &&
// RUN: grep 'III = common' %t

@interface XX
int x; 
int one=1; 
int two = 2; 
@end

@protocol PPP
int ddd;
@end

@interface XX(CAT)
  char * III;
@end


int main( int argc, const char *argv[] ) {
    return x+one+two;
}

