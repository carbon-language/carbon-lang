#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void srand(unsigned x) {}
double exp(double x) { return 0; }
double log(double x) { return 0; }
double sqrt(double x) { return 0; }
void exit(int x) {}
int puts(const char *x) { return 0; }
void __main() {}
int atoi(const char*x) { return 1; }
char *fgets(char*Ptr, int x, FILE*F) { return Ptr; }
int fclose(FILE*F) { return 0; }
FILE *fopen(const char *n, const char*x) { return malloc(sizeof(FILE)); }
int fflush(FILE *F) { return 0; }
size_t fwrite(const void* str, size_t N, size_t n, FILE *F) { return N; }
void *memset(void *P, int X, size_t N) { return P; }
char *strcpy(char*Str1, const char *Str) { return Str1; }
#undef putchar
int putchar(int N) { return N; }
int putc(int c, FILE *stream) { return c; }
int fputc(int c, FILE *stream) { return c; }
int fgetc(FILE *S) { return 0; }
int getc(FILE *S) { return 0; }
size_t fread(void *ptr, size_t size, size_t nmemb, FILE *stream) { return 0; }
int fseek(FILE *stream, long offset, int whence) { return 0; }
int fputs(const char *s, FILE *stream) { return 0; }
void rewind(FILE*F) { }
int fileno(FILE *stream) { return 1; }
char *ttyname(int desc) { return 0; }
long sysconf(int name) { return 0; }
char *tmpnam(char *s) { return s; }

void *calloc(size_t A, size_t B) { return malloc(A*B); }
void *realloc(void *ptr, size_t N) { return ptr; } 
const char *strerror(int N) { return 0; }
int unlink(const char *path) { return 0; }
void perror(const char *err) {}
char *strrchr(const char *S, int C) { return (char*)S; }
int memcmp(const char *A, const char *B, size_t N) { return 0; }
ssize_t read(int fildes, void *buf, size_t nbyte) { return nbyte; }
int close(int FD) { return 0; }
int rename(const char *oldpath, const char *newpath) { return 0; }
ssize_t write(int fd, const void *buf, size_t count) { return 0; }
pid_t getpid(void) { return 0; }
pid_t getppid(void) { return 0; }
void setbuf(FILE *stream, char *buf) {}
int isatty(int desc) { return 0; }


#include <sys/times.h>
clock_t times(struct tms *buf) { return 0; }


#include <setjmp.h>
int setjmp(jmp_buf env) { return 0; }
void longjmp(jmp_buf env, int val) {}
int kill(pid_t pid, int sig) { return 0; }
int system(const char *string) { return 0; }
char *getenv(const char *name) { return 0; }
typedef void (*sighandler_t)(int);

sighandler_t signal(int signum, sighandler_t handler) { return handler; }




char *strchr(const char *s, int c) { return (char*)s; }
int strcmp(const char *s1, const char *s2) { return 0; }
int strncmp(const char *s1, const char *s2, size_t n) { return 0; }
char *strncpy(char *s1, const char *s2, size_t n) { return s1; }
char *strpbrk(const char *s, const char *accept) { return (char*)s; }
char *strncat(char *dest, const char *src, size_t n) { return dest; }


long clock() { return 0; }
char *ctime(const time_t *timep) { return 0; }
time_t time(time_t *t) { return *t = 0; }

double sin(double x) { return x; }
double cos(double x) { return x; }
double atan(double x) { return x; }
double pow(double x, double y) { return x; }
int tolower(int x) { return x; }
int toupper(int x) { return x; }

