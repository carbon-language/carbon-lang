#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#undef assert
#define assert(X)


/* In the current implementation, each slab in the pool has NODES_PER_SLAB
 * nodes unless the isSingleArray flag is set in which case it contains a
 * single array of size ArraySize. Small arrays (size <= NODES_PER_SLAB) are
 * still allocated in the slabs of size NODES_PER_SLAB
 */
#define NODES_PER_SLAB 512 

typedef struct PoolTy {
  void    *Data;
  unsigned NodeSize;
  unsigned FreeablePool; /* Set to false if the memory from this pool cannot be
			    freed before destroy*/
  
} PoolTy;

/* PoolSlab Structure - Hold NODES_PER_SLAB objects of the current node type.
 *   Invariants: FirstUnused <= LastUsed+1
 */
typedef struct PoolSlab {
  unsigned FirstUnused;     /* First empty node in slab    */
  int LastUsed;             /* Last allocated node in slab */
  struct PoolSlab *Next;
  unsigned char AllocatedBitVector[NODES_PER_SLAB/8];
  unsigned char StartOfAllocation[NODES_PER_SLAB/8];

  unsigned isSingleArray;   /* If this slab is used for exactly one array */
  /* The array is allocated from the start to the end of the slab */
  unsigned ArraySize;       /* The size of the array allocated */ 

  char Data[1];   /* Buffer to hold data in this slab... variable sized */

} PoolSlab;

#define NODE_ALLOCATED(POOLSLAB, NODENUM) \
   ((POOLSLAB)->AllocatedBitVector[(NODENUM) >> 3] & (1 << ((NODENUM) & 7)))
#define MARK_NODE_ALLOCATED(POOLSLAB, NODENUM) \
   (POOLSLAB)->AllocatedBitVector[(NODENUM) >> 3] |= 1 << ((NODENUM) & 7)
#define MARK_NODE_FREE(POOLSLAB, NODENUM) \
   (POOLSLAB)->AllocatedBitVector[(NODENUM) >> 3] &= ~(1 << ((NODENUM) & 7))
#define ALLOCATION_BEGINS(POOLSLAB, NODENUM) \
   ((POOLSLAB)->StartOfAllocation[(NODENUM) >> 3] & (1 << ((NODENUM) & 7)))
#define SET_START_BIT(POOLSLAB, NODENUM) \
   (POOLSLAB)->StartOfAllocation[(NODENUM) >> 3] |= 1 << ((NODENUM) & 7)
#define CLEAR_START_BIT(POOLSLAB, NODENUM) \
   (POOLSLAB)->StartOfAllocation[(NODENUM) >> 3] &= ~(1 << ((NODENUM) & 7))


/* poolinit - Initialize a pool descriptor to empty
 */
void poolinit(PoolTy *Pool, unsigned NodeSize) {
  if (!Pool) {
    printf("Null pool pointer passed into poolinit!\n");
    exit(1);
  }

  Pool->NodeSize = NodeSize;
  Pool->Data = 0;

  Pool->FreeablePool = 1;

}

void poolmakeunfreeable(PoolTy *Pool) {
  if (!Pool) {
    printf("Null pool pointer passed in to poolmakeunfreeable!\n");
    exit(1);
  }

  Pool->FreeablePool = 0;
}

/* pooldestroy - Release all memory allocated for a pool
 */
void pooldestroy(PoolTy *Pool) {
  PoolSlab *PS;
  if (!Pool) {
    printf("Null pool pointer passed in to pooldestroy!\n");
    exit(1);
  }

  PS = (PoolSlab*)Pool->Data;
  while (PS) {
    PoolSlab *Next = PS->Next;
    free(PS);
    PS = Next;
  }
}

static void *FindSlabEntry(PoolSlab *PS, unsigned NodeSize) {
  /* Loop through all of the slabs looking for one with an opening */
  for (; PS; PS = PS->Next) {

    /* If the slab is a single array, go on to the next slab */
    /* Don't allocate single nodes in a SingleArray slab */
    if (PS->isSingleArray) 
      continue;

    /* Check to see if there are empty entries at the end of the slab... */
    if (PS->LastUsed < NODES_PER_SLAB-1) {
      /* Mark the returned entry used */
      MARK_NODE_ALLOCATED(PS, PS->LastUsed+1);
      SET_START_BIT(PS, PS->LastUsed+1);

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

      MARK_NODE_ALLOCATED(PS, Idx);
      SET_START_BIT(PS, Idx);

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
  unsigned NodeSize;
  PoolSlab *PS;
  void *Result;

  if (!Pool) {
    printf("Null pool pointer passed in to poolalloc!\n");
    exit(1);
  }
  
  NodeSize = Pool->NodeSize;
  // Return if this pool has size 0
  if (NodeSize == 0)
    return 0;

  PS = (PoolSlab*)Pool->Data;

  if ((Result = FindSlabEntry(PS, NodeSize)))
    return Result;

  /* Otherwise we must allocate a new slab and add it to the list */
  PS = (PoolSlab*)malloc(sizeof(PoolSlab)+NodeSize*NODES_PER_SLAB-1);

  if (!PS) {
    printf("poolalloc: Could not allocate memory!");
    exit(1);
  }

  /* Initialize the slab to indicate that the first element is allocated */
  PS->FirstUnused = 1;
  PS->LastUsed = 0;
  /* This is not a single array */
  PS->isSingleArray = 0;
  PS->ArraySize = 0;
  
  MARK_NODE_ALLOCATED(PS, 0);
  SET_START_BIT(PS, 0);

  /* Add the slab to the list... */
  PS->Next = (PoolSlab*)Pool->Data;
  Pool->Data = PS;
  return &PS->Data[0];
}

void poolfree(PoolTy *Pool, char *Node) {
  unsigned NodeSize, Idx;
  PoolSlab *PS;
  PoolSlab **PPS;
  unsigned idxiter;

  if (!Pool) {
    printf("Null pool pointer passed in to poolfree!\n");
    exit(1);
  }

  NodeSize = Pool->NodeSize;

  // Return if this pool has size 0
  if (NodeSize == 0)
    return;

  PS = (PoolSlab*)Pool->Data;
  PPS = (PoolSlab**)&Pool->Data;

  /* Search for the slab that contains this node... */
  while (&PS->Data[0] > Node || &PS->Data[NodeSize*NODES_PER_SLAB-1] < Node) {
    if (!PS) { 
      printf("poolfree: node being free'd not found in allocation pool specified!\n");
      exit(1);
    }

    PPS = &PS->Next;
    PS = PS->Next;
  }

  /* PS now points to the slab where Node is */

  Idx = (Node-&PS->Data[0])/NodeSize;
  assert(Idx < NODES_PER_SLAB && "Pool slab searching loop broken!");

  if (PS->isSingleArray) {

    /* If this slab is a SingleArray */

    if (Idx != 0) {
      printf("poolfree: Attempt to free middle of allocated array\n");
      exit(1);
    }
    if (!NODE_ALLOCATED(PS,0)) {
      printf("poolfree: Attempt to free node that is already freed\n");
      exit(1);
    }
    /* Mark this SingleArray slab as being free by just marking the first
       entry as free*/
    MARK_NODE_FREE(PS, 0);
  } else {
    
    /* If this slab is not a SingleArray */
    
    if (!ALLOCATION_BEGINS(PS, Idx)) { 
      printf("poolfree: Attempt to free middle of allocated array\n");
    }

    /* Free the first node */
    if (!NODE_ALLOCATED(PS, Idx)) {
      printf("poolfree: Attempt to free node that is already freed\n");
      exit(1); 
    }
    CLEAR_START_BIT(PS, Idx);
    MARK_NODE_FREE(PS, Idx);
    
    // Free all nodes 
    idxiter = Idx + 1;
    while (idxiter < NODES_PER_SLAB && (!ALLOCATION_BEGINS(PS,idxiter)) && 
	   (NODE_ALLOCATED(PS, idxiter))) {
      MARK_NODE_FREE(PS, idxiter);
      ++idxiter;
    }

    /* Update the first free field if this node is below the free node line */
    if (Idx < PS->FirstUnused) PS->FirstUnused = Idx;
    
    /* If we are not freeing the last element in a slab... */
    if (idxiter - 1 != PS->LastUsed) {
      return;
    }

    /* Otherwise we are freeing the last element in a slab... shrink the
     * LastUsed marker down to last used node.
     */
    PS->LastUsed = Idx;
    do {
      --PS->LastUsed;
      /* Fixme, this should scan the allocated array an entire byte at a time 
       * for performance!
       */
    } while (PS->LastUsed >= 0 && (!NODE_ALLOCATED(PS, PS->LastUsed)));
    
    assert(PS->FirstUnused <= PS->LastUsed+1 &&
	   "FirstUnused field was out of date!");
  }
    
  /* Ok, if this slab is empty, we unlink it from the of slabs and either move
   * it to the head of the list, or free it, depending on whether or not there
   * is already an empty slab at the head of the list.
   */
  /* Do this only if the pool is freeable */
  if (Pool->FreeablePool) {
    if (PS->isSingleArray) {
      /* If it is a SingleArray, just free it */
      *PPS = PS->Next;
      free(PS);
    } else if (PS->LastUsed == -1) {   /* Empty slab? */
      PoolSlab *HeadSlab;
      *PPS = PS->Next;   /* Unlink from the list of slabs... */
      
      HeadSlab = (PoolSlab*)Pool->Data;
      if (HeadSlab && HeadSlab->LastUsed == -1){/*List already has empty slab?*/
	free(PS);                               /*Free memory for slab */
      } else {
	PS->Next = HeadSlab;                    /*No empty slab yet, add this*/
	Pool->Data = PS;                        /*one to the head of the list */
      }
    }
  } else {
    /* Pool is not freeable for safety reasons */
    /* Leave it in the list of PoolSlabs as an empty PoolSlab */
    if (!PS->isSingleArray)
      if (PS->LastUsed == -1) {
	PS->FirstUnused = 0;
	
	/* Do not free the pool, but move it to the head of the list if there is
	   no empty slab there already */
	PoolSlab *HeadSlab;
	HeadSlab = (PoolSlab*)Pool->Data;
	if (HeadSlab && HeadSlab->LastUsed != -1) {
	  PS->Next = HeadSlab;
	  Pool->Data = PS;
	}
      }
  }
}

/* The poolallocarray version of FindSlabEntry */
static void *FindSlabEntryArray(PoolSlab *PS, unsigned NodeSize, 
				unsigned Size) {
  unsigned i;

  /* Loop through all of the slabs looking for one with an opening */
  for (; PS; PS = PS->Next) {
    
    /* For large array allocation */
    if (Size > NODES_PER_SLAB) {
      /* If this slab is a SingleArray that is free with size > Size, use it */
      if (PS->isSingleArray && !NODE_ALLOCATED(PS,0) && PS->ArraySize >= Size) {
	/* Allocate the array in this slab */
	MARK_NODE_ALLOCATED(PS,0); /* In a single array, only the first node
				      needs to be marked */
	return &PS->Data[0];
      } else
	continue;
    } else if (PS->isSingleArray)
      continue; /* Do not allocate small arrays in SingleArray slabs */

    /* For small array allocation */
    /* Check to see if there are empty entries at the end of the slab... */
    if (PS->LastUsed < NODES_PER_SLAB-Size) {
      /* Mark the returned entry used and set the start bit*/
      SET_START_BIT(PS, PS->LastUsed + 1);
      for (i = PS->LastUsed + 1; i <= PS->LastUsed + Size; ++i)
	MARK_NODE_ALLOCATED(PS, i);

      /* If we are allocating out the first unused field, bump its index also */
      if (PS->FirstUnused == PS->LastUsed+1)
        PS->FirstUnused += Size;

      /* Increment LastUsed */
      PS->LastUsed += Size;

      /* Return the entry */
      return &PS->Data[0] + (PS->LastUsed - Size + 1) * NodeSize;
    }

    /* If not, check to see if this node has a declared "FirstUnused" value
     * starting which Size nodes can be allocated
     */
    if (PS->FirstUnused < NODES_PER_SLAB - Size + 1 &&
	(PS->LastUsed < PS->FirstUnused || 
	 PS->LastUsed - PS->FirstUnused >= Size)) {
      unsigned Idx = PS->FirstUnused, foundArray;
      
      /* Check if there is a continuous array of Size nodes starting 
	 FirstUnused */
      foundArray = 1;
      for (i = Idx; (i < Idx + Size) && foundArray; ++i)
	if (NODE_ALLOCATED(PS, i))
	  foundArray = 0;

      if (foundArray) {
	/* Successfully allocate starting from the first unused node */
	SET_START_BIT(PS, Idx);
	for (i = Idx; i < Idx + Size; ++i)
	  MARK_NODE_ALLOCATED(PS, i);
	
	PS->FirstUnused += Size;
	while (PS->FirstUnused < NODES_PER_SLAB &&
               NODE_ALLOCATED(PS, PS->FirstUnused)) {
	  ++PS->FirstUnused;
	}
	return &PS->Data[0] + Idx*NodeSize;
      }
      
    }
  }

  /* No empty nodes available, must grow # slabs! */
  return 0;
}

char* poolallocarray(PoolTy* Pool, unsigned Size) {
  unsigned NodeSize;
  PoolSlab *PS;
  void *Result;
  unsigned i;

  if (!Pool) {
    printf("Null pool pointer passed to poolallocarray!\n");
    exit(1);
  }

  NodeSize = Pool->NodeSize;

  // Return if this pool has size 0
  if (NodeSize == 0)
    return 0;

  PS = (PoolSlab*)Pool->Data;

  if ((Result = FindSlabEntryArray(PS, NodeSize,Size)))
    return Result;

  /* Otherwise we must allocate a new slab and add it to the list */
  if (Size > NODES_PER_SLAB) {
    /* Allocate a new slab of size Size */
    PS = (PoolSlab*)malloc(sizeof(PoolSlab)+NodeSize*Size-1);
    if (!PS) {
      printf("poolallocarray: Could not allocate memory!\n");
      exit(1);
    }
    PS->isSingleArray = 1;
    PS->ArraySize = Size;
    MARK_NODE_ALLOCATED(PS, 0);
  } else {
    PS = (PoolSlab*)malloc(sizeof(PoolSlab)+NodeSize*NODES_PER_SLAB-1);
    if (!PS) {
      printf("poolallocarray: Could not allocate memory!\n");
      exit(1);
    }

    /* Initialize the slab to indicate that the first element is allocated */
    PS->FirstUnused = Size;
    PS->LastUsed = Size - 1;
    
    PS->isSingleArray = 0;
    PS->ArraySize = 0;

    SET_START_BIT(PS, 0);
    for (i = 0; i < Size; ++i) {
      MARK_NODE_ALLOCATED(PS, i);
    }
  }

  /* Add the slab to the list... */
  PS->Next = (PoolSlab*)Pool->Data;
  Pool->Data = PS;
  return &PS->Data[0];
}
