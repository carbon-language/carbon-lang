// RUN: clang -triple=i686-apple-darwin9 -fnext-runtime -emit-llvm -o %t %s &&
// RUN: grep -e "\^i" %t | count 1 &&
// RUN: grep -e "\[0i\]" %t | count 1

int main()
{
  int n;
  
  const char * inc = @encode(int[]);
  const char * vla = @encode(int[n]);
}
