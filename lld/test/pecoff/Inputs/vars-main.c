__declspec(dllimport) int var;
__declspec(dllimport) int fn(void);
__declspec(dllimport) int _name_with_underscore(void);

int main() {
  return var + fn() + _name_with_underscore();
}
