/* FIXME: this testcase should be automated! */

#include <stdio.h>

typedef struct Tree_struct {
  int data;
  struct Tree_struct *left, *right;
} Tree;

static Tree T1, T2, T3, T4, T5, T6, T7;
static Tree *Root, *ANode;
static int  N = 4107;

/* forces *Tb->right to be collapsed */
void makeMore(Tree* Ta, Tree* Tb)
{
  Ta->left  = &T1;
  Ta->right = &T2;
  Tb->left  = &T4;
  /* Tb->right = &T5; */
  Tb->right = (Tree*) (((char*) &T5) + 5); /* point to fifth byte of T5 */
}

/* multiple calls to this should force globals to be merged in TD graph
 * but not in globals graph
 */
void makeData(Tree* Ta)
{
  static int N = 101;
  Ta->data = N;
}

void makeRoots()
{
  T1.left = &T2;
  makeMore(&T1, &T3);
}

/* BU graph shows T1.left->{T2}, but TD graph should show T1.left->{T1,T2,T6,H}
 * and T.right->{T1,T2,T6,H} */
void makeAfter1()
{
  T1.left = &T2;
}

/* BU graph shows:
 *      T2.right->{H}, H.left->{T6}; H.right->{T2}, T3.left<->T7.left
 * 
 * TD graph and GlobalsGraph should show:
 *      T2.right->{T1,T2,T6,H}
 *      H.left->{T1,T2,T6,H}; H.right->{T1,T2,T6,H}.
 *      T3.left->{T4,T7}, T3.right->{T4,T7}, T7.left->{T3}
 */
void makeAfter2()
{
  Tree* newT  = (Tree*) malloc(sizeof(Tree));
  T2.right    = newT;                   /* leaked: do not access T2 in main */
  newT->left  = &T6;
  newT->right = &T2;

  T3.left     = &T7;
  T7.left     = &T3;
}

/* BU and TD graphs should have no reachable globals, forcing callers and
 * callees to get all globals from GlobalsGraph
 */
void makePass()
{
  makeAfter1();
  makeAfter2();
}

int main()
{
  makeRoots();
  T3.right = &T4;
  makeData(&T3);
  makeData(&T5);
  makePass();
  printf("T3.data = %d\n", T3.data);
}
