__declspec(dllimport) int var;
__declspec(dllimport) int fn(void);

int main() {
  return var + fn();
}
