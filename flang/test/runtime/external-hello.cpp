#include "../../runtime/io-api.h"
#include "../../runtime/main.h"
#include "../../runtime/stop.h"
#include <cstring>

using namespace Fortran::runtime::io;

int main(int argc, const char *argv[], const char *envp[]) {
  static const char *format{"(12HHELLO, WORLD)"};
  RTNAME(ProgramStart)(argc, argv, envp);
  auto *io{IONAME(BeginExternalFormattedOutput)(format, std::strlen(format))};
  IONAME(EndIoStatement)(io);
  RTNAME(ProgramEndStatement)();
  return 0;
}
