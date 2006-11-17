#include <stdio.h>
void abort(void);

/* This is used by the `assert' macro.  */
void
__eprintf (const char *string, const char *expression,
           unsigned int line, const char *filename)
{
  fprintf (stderr, string, expression, line, filename);
  fflush (stderr);
  abort ();
}

