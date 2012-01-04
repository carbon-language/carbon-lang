#pragma clang system_header

typedef struct _FILE FILE;
extern FILE *stdin;
int fscanf(FILE *restrict stream, const char *restrict format, ...);

// Note, on some platforms errno macro gets replaced with a function call.
extern int errno;

unsigned long strlen(const char *);
