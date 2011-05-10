#include <stdio.h>

extern int AddElement (char *value);
extern char *GetElement (int idx);
extern void *GetArray();

int
main ()
{

  int idx = AddElement ("some string");
  void *array_token = GetArray();

  char *string = GetElement (0); // Set breakpoint here, then do 'expr (NSArray*)array_token'.
  if (string)
    printf ("This: %s.\n", string);

  return 0;
}  
