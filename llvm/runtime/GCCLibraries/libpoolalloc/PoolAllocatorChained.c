#include <assert.h>
#include <stdlib.h>

#undef assert
#define assert(X)


#define NODES_PER_SLAB 512

typedef struct PoolTy {
  void    *Data;
  unsigned NodeSize;
} PoolTy;

/* PoolSlab Structure - Hold NODES_PER_SLAB objects of the current node type.
 *   Invariants: FirstUnused <= LastUsed+1
 */
typedef struct PoolSlab {
  unsigned FirstUnused;     /* First empty node in slab    */
  int LastUsed;             /* Last allocated node in slab */
  struct PoolSlab *Next;
  unsigned char AllocatedBitVector[NODES_PER_SLAB/8];
  char Data[1];   /* Buffer to hold data in this slab... variable sized */
} PoolSlab;

#define NODE_ALLOCATED(POOLSLAB, NODENUM) \
   ((POOLSLAB)->AllocatedBitVector[(NODENUM) >> 3] & (1 << ((NODENUM) & 7)))
#define MARK_NODE_ALLOCATED(POOLSLAB, NODENUM) \
   (POOLSLAB)->AllocatedBitVector[(NODENUM) >> 3] |= 1 << ((NODENUM) & 7)
#define MARK_NODE_FREE(POOLSLAB, NODENUM) \
   (POOLSLAB)->AllocatedBitVector[(NODENUM) >> 3] &= ~(1 << ((NODENUM) & 7))


/* poolinit - Initialize a pool descriptor to empty
 */
void poolinit(PoolTy *Pool, unsigned Size) {
  assert(Pool && "Null pool pointer passed in!");

  Pool->NodeSize = Size;
  Pool->Data = 0;
}

/* pooldestroy - Release all memory allocated for a pool
 */
void pooldestroy(PoolTy *Pool) {
  PoolSlab *PS = (PoolSlab*)Pool->Data;
  while (PS) {
    PoolSlab *Next = PS->Next;
    free(PS);
    PS = Next;
  }
}

static void *FindSlabEntry(PoolSlab *PS, unsigned NodeSize) {
  /* Loop through all of the slabs looking for one with an opening */
  for (; PS; PS = PS->Next) {
    /* Check to see if there are empty entries at the end of the slab... */
    if (PS->LastUsed < NODES_PER_SLAB-1) {
      /* Mark the returned entry used */
      MARK_NODE_ALLOCATED(PS, PS->LastUsed+1);

      /* If we are allocating out the first unused field, bump its index also */
      if (PS->FirstUnused == PS->LastUsed+1)
        PS->FirstUnused++;

      /* Return the entry, increment LastUsed field. */
      return &PS->Data[0] + ++PS->LastUsed * NodeSize;
    }

    /* If not, check to see if this node has a declared "FirstUnused" value that
     * is less than the number of nodes allocated...
     */
    if (PS->FirstUnused < NODES_PER_SLAB) {
      /* Successfully allocate out the first unused node */
      unsigned Idx = PS->FirstUnused;
      
      /* Increment FirstUnused to point to the new first unused value... */
      do {
        ++PS->FirstUnused;
      } while (PS->FirstUnused < NODES_PER_SLAB &&
               NODE_ALLOCATED(PS, PS->FirstUnused));

      return &PS->Data[0] + Idx*NodeSize;
    }
  }

  /* No empty nodes available, must grow # slabs! */
  return 0;
}

char *poolalloc(PoolTy *Pool) {
  unsigned NodeSize = Pool->NodeSize;
  PoolSlab *PS = (PoolSlab*)Pool->Data;
  void *Result;

  if ((Result = FindSlabEntry(PS, NodeSize)))
    return Result;

  /* Otherwise we must allocate a new slab and add it to the list */
  PS = (PoolSlab*)malloc(sizeof(PoolSlab)+NodeSize*NODES_PER_SLAB-1);

  /* Initialize the slab to indicate that the first element is allocated */
  PS->FirstUnused = 1;
  PS->LastUsed = 0;
  PS->AllocatedBitVector[0] = 1;

  /* Add the slab to the list... */
  PS->Next = (PoolSlab*)Pool->Data;
  Pool->Data = PS;
  return &PS->Data[0];
}

void poolfree(PoolTy *Pool, char *Node) {
  unsigned NodeSize = Pool->NodeSize, Idx;
  PoolSlab *PS = (PoolSlab*)Pool->Data;
  PoolSlab **PPS = (PoolSlab**)&Pool->Data;

  /* Seach for the slab that contains this node... */
  while (&PS->Data[0] > Node || &PS->Data[NodeSize*NODES_PER_SLAB] < Node) {
    assert(PS && "free'd node not found in allocation pool specified!");
    PPS = &PS->Next;
    PS = PS->Next;
  }

  Idx = (Node-&PS->Data[0])/NodeSize;
  assert(Idx < NODES_PER_SLAB && "Pool slab searching loop broken!");

  /* Update the first free field if this node is below the free node line */
  if (Idx < PS->FirstUnused) PS->FirstUnused = Idx;
  
  /* If we are not freeing the last element in a slab... */
  if (Idx != PS->LastUsed) {
    MARK_NODE_FREE(PS, Idx);
    return;
  } 

  /* Otherwise we are freeing the last element in a slab... shrink the
   * LastUsed marker down to last used node.
   */
  do {
    --PS->LastUsed;
    /* Fixme, this should scan the allocated array an entire byte at a time for
     * performance!
     */
  } while (PS->LastUsed >= 0 && (!NODE_ALLOCATED(PS, PS->LastUsed)));
  
  assert(PS->FirstUnused <= PS->LastUsed+1 &&
         "FirstUnused field was out of date!");
    
  /* Ok, if this slab is empty, we unlink it from the of slabs and either move
   * it to the head of the list, or free it, depending on whether or not there
   * is already an empty slab at the head of the list.
   */
  if (PS->LastUsed == -1) {   /* Empty slab? */
    PoolSlab *HeadSlab;
    *PPS = PS->Next;   /* Unlink from the list of slabs... */

    HeadSlab = (PoolSlab*)Pool->Data;
    if (HeadSlab && HeadSlab->LastUsed == -1){/* List already has empty slab? */
      free(PS);                               /* Free memory for slab */
    } else {
      PS->Next = HeadSlab;                    /* No empty slab yet, add this */
      Pool->Data = PS;                        /* one to the head of the list */
    }
  }
}
