#include <stdio.h>

typedef struct Tree_struct {
  int data;
  struct Tree_struct *left, *right;
} Tree;

static Tree T1, T2, T3, T4, T5;
static Tree *Root, *ANode;
static int  N = 4107;

/* forces *Tb->right to be collapsed */
void makeMore(Tree* Ta, Tree* Tb)
{
  Ta->left  = &T1;
  Ta->right = &T2;
  Tb->left  = &T4;
  Tb->right = (Tree*) (((char*) &T5) + 5); /* point to fifth byte of T5 */
}

void makeRoots()
{
  T1.left = &T2;
  makeMore(&T1, &T3);
}

int main()
{
  makeRoots();
  T3.right = &T4;
  printf("T3.data = %d\n", T3.data);
}
