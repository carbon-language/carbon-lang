// RUN: clang -parse-ast-check %s
int main() {
  char *s;
  s = (char []){"whatever"}; 
}
