/*===---- mm_malloc.h - Allocating and Freeing Aligned Memory Blocks -------===
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __MM_MALLOC_H
#define __MM_MALLOC_H

#include <errno.h>
#include <stdlib.h>

static __inline__ void *__attribute__((__always_inline__, __nodebug__))
_mm_malloc(size_t size, size_t align)
{
  if (align & (align - 1)) {
    errno = EINVAL;
    return 0;
  }

  if (!size)
    return 0;

  if (align < 2 * sizeof(void *))
    align = 2 * sizeof(void *);

  void *mallocedMemory = malloc(size + align);
  if (!mallocedMemory)
    return 0;

  void *alignedMemory =
    (void *)(((size_t)mallocedMemory + align) & ~((size_t)align - 1));
  ((void **)alignedMemory)[-1] = mallocedMemory;

  return alignedMemory;
}

static __inline__ void __attribute__((__always_inline__, __nodebug__))
_mm_free(void *p)
{
  if (p)
    free(((void **)p)[-1]);
}

#endif /* __MM_MALLOC_H */
