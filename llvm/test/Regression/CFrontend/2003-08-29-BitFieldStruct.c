typedef enum { FALSE, TRUE } flagT;

struct Word
{
  short bar;
  short baz;
  flagT final:1;
  short quux;
} *word_limit;

void foo ()
{
  word_limit->final = (word_limit->final && word_limit->final);
}
