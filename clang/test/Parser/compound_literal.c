// RUN: clang -parse-ast -verify %s
int main() {
  char *s;
  s = (char []){"whatever"}; 
}
