#if !defined(__APPLE__)

#include <cstddef>
#include <cstdint>

#else

typedef __SIZE_TYPE__ size_t;
#define __SSIZE_TYPE__                                                         \
  __typeof__(_Generic((__SIZE_TYPE__)0, unsigned long long int                 \
                      : (long long int)0, unsigned long int                    \
                      : (long int)0, unsigned int                              \
                      : (int)0, unsigned short                                 \
                      : (short)0, unsigned char                                \
                      : (signed char)0))
typedef __SSIZE_TYPE__ ssize_t;

typedef unsigned long long uint64_t;

#endif

#include "config.h"

#ifdef HAVE_ELF_H
#include <elf.h>
#endif

// Save all registers while keeping 16B stack alignment
#define SAVE_ALL                                                               \
  "push %%rax\n"                                                               \
  "push %%rbx\n"                                                               \
  "push %%rcx\n"                                                               \
  "push %%rdx\n"                                                               \
  "push %%rdi\n"                                                               \
  "push %%rsi\n"                                                               \
  "push %%rbp\n"                                                               \
  "push %%r8\n"                                                                \
  "push %%r9\n"                                                                \
  "push %%r10\n"                                                               \
  "push %%r11\n"                                                               \
  "push %%r12\n"                                                               \
  "push %%r13\n"                                                               \
  "push %%r14\n"                                                               \
  "push %%r15\n"                                                               \
  "sub $8, %%rsp\n"

// Mirrors SAVE_ALL
#define RESTORE_ALL                                                            \
  "add $8, %%rsp\n"                                                            \
  "pop %%r15\n"                                                                \
  "pop %%r14\n"                                                                \
  "pop %%r13\n"                                                                \
  "pop %%r12\n"                                                                \
  "pop %%r11\n"                                                                \
  "pop %%r10\n"                                                                \
  "pop %%r9\n"                                                                 \
  "pop %%r8\n"                                                                 \
  "pop %%rbp\n"                                                                \
  "pop %%rsi\n"                                                                \
  "pop %%rdi\n"                                                                \
  "pop %%rdx\n"                                                                \
  "pop %%rcx\n"                                                                \
  "pop %%rbx\n"                                                                \
  "pop %%rax\n"

// Anonymous namespace covering everything but our library entry point
namespace {

#if defined(__APPLE__)

uint64_t __write(uint64_t fd, const void *buf, uint64_t count) {
  uint64_t ret;
  const long write = 0x2000004;
  __asm__ __volatile__("syscall;\n"
                       "movq %%rax, %0;\n"
                       : "=g"(ret)
                       : /* rax */ "a"(write), /* rdi */ "D"(fd),
                         /* rsi */ "S"(buf), /* rdx */ "d"(count)
                       : "memory");
  return ret;
}

#else

// We use a stack-allocated buffer for string manipulation in many pieces of
// this code, including the code that prints each line of the fdata file. This
// buffer needs to accomodate large function names, but shouldn't be arbitrarily
// large (dynamically allocated) for simplicity of our memory space usage.
constexpr uint32_t BufSize = 10240;

// Declare some syscall wrappers we use throughout this code to avoid linking
// against system libc.
uint64_t __open(const char *pathname, uint64_t flags, uint64_t mode) {
  uint64_t ret;
  __asm__ __volatile__("movq $2, %%rax\n"
                       "syscall"
                       : "=a"(ret)
                       : "D"(pathname), "S"(flags), "d"(mode)
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

uint64_t __write(uint64_t fd, const void *buf, uint64_t count) {
  uint64_t ret;
  __asm__ __volatile__("movq $1, %%rax\n"
                       "syscall\n"
                       : "=a"(ret)
                       : "D"(fd), "S"(buf), "d"(count)
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

uint64_t __lseek(uint64_t fd, uint64_t pos, uint64_t whence) {
  uint64_t ret;
  __asm__ __volatile__("movq $8, %%rax\n"
                       "syscall\n"
                       : "=a"(ret)
                       : "D"(fd), "S"(pos), "d"(whence)
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

int __close(uint64_t fd) {
  uint64_t ret;
  __asm__ __volatile__("movq $3, %%rax\n"
                       "syscall\n"
                       : "=a"(ret)
                       : "D"(fd)
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

int __madvise(void *addr, size_t length, int advice) {
  int ret;
  __asm__ __volatile__("movq $28, %%rax\n"
                       "syscall\n"
                       : "=a"(ret)
                       : "D"(addr), "S"(length), "d"(advice)
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

/* Length of the entries in `struct utsname' is 65.  */
#define _UTSNAME_LENGTH 65

struct utsname {
  char sysname[_UTSNAME_LENGTH];  /* Operating system name (e.g., "Linux") */
  char nodename[_UTSNAME_LENGTH]; /* Name within "some implementation-defined
                      network" */
  char release[_UTSNAME_LENGTH]; /* Operating system release (e.g., "2.6.28") */
  char version[_UTSNAME_LENGTH]; /* Operating system version */
  char machine[_UTSNAME_LENGTH]; /* Hardware identifier */
  char domainname[_UTSNAME_LENGTH]; /* NIS or YP domain name */
};

int __uname(struct utsname *buf) {
  int ret;
  __asm__ __volatile__("movq $63, %%rax\n"
                       "syscall\n"
                       : "=a"(ret)
                       : "D"(buf)
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

struct timespec {
  uint64_t tv_sec;  /* seconds */
  uint64_t tv_nsec; /* nanoseconds */
};

uint64_t __nanosleep(const timespec *req, timespec *rem) {
  uint64_t ret;
  __asm__ __volatile__("movq $35, %%rax\n"
                       "syscall\n"
                       : "=a"(ret)
                       : "D"(req), "S"(rem)
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

int64_t __fork() {
  uint64_t ret;
  __asm__ __volatile__("movq $57, %%rax\n"
                       "syscall\n"
                       : "=a"(ret)
                       :
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

void *__mmap(uint64_t addr, uint64_t size, uint64_t prot, uint64_t flags,
             uint64_t fd, uint64_t offset) {
  void *ret;
  register uint64_t r8 asm("r8") = fd;
  register uint64_t r9 asm("r9") = offset;
  register uint64_t r10 asm("r10") = flags;
  __asm__ __volatile__("movq $9, %%rax\n"
                       "syscall\n"
                       : "=a"(ret)
                       : "D"(addr), "S"(size), "d"(prot), "r"(r10), "r"(r8),
                         "r"(r9)
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

int __mprotect(void *addr, size_t len, int prot) {
  int ret;
  __asm__ __volatile__("movq $10, %%rax\n"
                       "syscall\n"
                       : "=a"(ret)
                       : "D"(addr), "S"(len), "d"(prot)
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

uint64_t __munmap(void *addr, uint64_t size) {
  uint64_t ret;
  __asm__ __volatile__("movq $11, %%rax\n"
                       "syscall\n"
                       : "=a"(ret)
                       : "D"(addr), "S"(size)
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

uint64_t __getpid() {
  uint64_t ret;
  __asm__ __volatile__("movq $39, %%rax\n"
                       "syscall\n"
                       : "=a"(ret)
                       :
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

uint64_t __getppid() {
  uint64_t ret;
  __asm__ __volatile__("movq $110, %%rax\n"
                       "syscall\n"
                       : "=a"(ret)
                       :
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

uint64_t __exit(uint64_t code) {
  uint64_t ret;
  __asm__ __volatile__("movq $231, %%rax\n"
                       "syscall\n"
                       : "=a"(ret)
                       : "D"(code)
                       : "cc", "rcx", "r11", "memory");
  return ret;
}

// Helper functions for writing strings to the .fdata file. We intentionally
// avoid using libc names (lowercase memset) to make it clear it is our impl.

/// Write number Num using Base to the buffer in OutBuf, returns a pointer to
/// the end of the string.
char *intToStr(char *OutBuf, uint64_t Num, uint32_t Base) {
  const char *Chars = "0123456789abcdef";
  char Buf[21];
  char *Ptr = Buf;
  while (Num) {
    *Ptr++ = *(Chars + (Num % Base));
    Num /= Base;
  }
  if (Ptr == Buf) {
    *OutBuf++ = '0';
    return OutBuf;
  }
  while (Ptr != Buf) {
    *OutBuf++ = *--Ptr;
  }
  return OutBuf;
}

/// Copy Str to OutBuf, returns a pointer to the end of the copied string
char *strCopy(char *OutBuf, const char *Str, int32_t Size = BufSize) {
  while (*Str) {
    *OutBuf++ = *Str++;
    if (--Size <= 0)
      return OutBuf;
  }
  return OutBuf;
}

void memSet(char *Buf, char C, uint32_t Size) {
  for (int I = 0; I < Size; ++I)
    *Buf++ = C;
}

void *memCpy(void *Dest, const void *Src, size_t Len) {
  char *d = static_cast<char *>(Dest);
  const char *s = static_cast<const char *>(Src);
  while (Len--)
    *d++ = *s++;
  return Dest;
}

uint32_t strLen(const char *Str) {
  uint32_t Size = 0;
  while (*Str++)
    ++Size;
  return Size;
}

void reportError(const char *Msg, uint64_t Size) {
  __write(2, Msg, Size);
  __exit(1);
}

void assert(bool Assertion, const char *Msg) {
  if (Assertion)
    return;
  char Buf[BufSize];
  char *Ptr = Buf;
  Ptr = strCopy(Ptr, "Assertion failed: ");
  Ptr = strCopy(Ptr, Msg, BufSize - 40);
  Ptr = strCopy(Ptr, "\n");
  reportError(Buf, Ptr - Buf);
}

void reportNumber(const char *Msg, uint64_t Num, uint32_t Base) {
  char Buf[BufSize];
  char *Ptr = Buf;
  Ptr = strCopy(Ptr, Msg, BufSize - 23);
  Ptr = intToStr(Ptr, Num, Base);
  Ptr = strCopy(Ptr, "\n");
  __write(2, Buf, Ptr - Buf);
}

void report(const char *Msg) { __write(2, Msg, strLen(Msg)); }

/// 1B mutex accessed by lock xchg
class Mutex {
  volatile bool InUse{false};

public:
  bool acquire() {
    bool Result = true;
    asm volatile("lock; xchg %0, %1" : "+m"(InUse), "=r"(Result) : : "cc");
    return !Result;
  }
  void release() { InUse = false; }
};

/// RAII wrapper for Mutex
class Lock {
  Mutex &M;

public:
  Lock(Mutex &M) : M(M) {
    while (!M.acquire()) {
    }
  }
  ~Lock() { M.release(); }
};

inline uint64_t alignTo(uint64_t Value, uint64_t Align) {
  return (Value + Align - 1) / Align * Align;
}

#endif

} // anonymous namespace
