/*
 * Program: llc
 * 
 * Test Name: badfuncptr.c
 * 
 * Test Problem:
 *      Indirect call via function pointer is mishandled in reg. alloc.
 *      The indirect call address was allocated the same register as the
 *      first outgoing argument, so it was overwritten before the call.
 *
 * Test Resolution:
 *      In PhyRegAlloc.cpp, mark the live range for the indirect call
 *      address as having a Call Interference.  This has to be done
 *      as a special case since it may not be live after the call.
 *
 * Resolution Status:
 *      Fixed on 3/29/02 -- Adve.
 */
/* For copyright information, see olden_v1.0/COPYRIGHT */

#include <stdlib.h>
/* #include "hash.h" */
/*--------*/
/* hash.h */
/*--------*/
/* For copyright information, see olden_v1.0/COPYRIGHT */

#include "stdio.h"

typedef struct hash_entry {
  unsigned int key;
  void *entry;
  struct hash_entry *next;
} *HashEntry;

typedef struct hash {
  HashEntry *array;
  int (*mapfunc)(unsigned int);
  int size;
} *Hash;

Hash MakeHash(int size, int (*map)(unsigned int));
void *HashLookup(unsigned int key, Hash hash);
void HashInsert(void *entry,unsigned int key, Hash hash);
void HashDelete(unsigned int key, Hash hash);
/*--------*/
/* END hash.h */
/*--------*/

#define assert(num,a) if (!(a)) {printf("Assertion failure:%d in hash\n",num); exit(-1);}

void *HashLookup(unsigned int key, Hash hash)
{
  int j;
  HashEntry ent;
  
  j = (hash->mapfunc)(key);        /* 14% miss in hash->mapfunc */  
  assert(1,j>=0);
  assert(2,j<hash->size);
  for (ent = hash->array[j];       /* 17% miss in hash->array[j] */ /* adt_pf can't detect :( */
       ent &&                      /* 47% miss in ent->key */       /* adt_pf can detect :) */
           ent->key!=key; 
       ent=ent->next);             /* 8% miss in ent->next */       /* adt_pf can detect :) */
  if (ent) return ent->entry;
  return NULL;
}

/* essentially dummy main so testing does not fail */
int
main()
{
  printf("&HashLookup = 0x%p\n", HashLookup);
  return 0;
}
