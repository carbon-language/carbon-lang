/*===- StorageProxy.c - OS implementation of the caching interface --------===*\
 *                                                                            *
 * This file implements the interface that we will expect operating           *
 * systems to implement if they wish to support offline cachine.              *
 *                                                                            *
\*===----------------------------------------------------------------------===*/

#include "OSInterface.h"
#include "SysUtils.h"
#include "Config/fcntl.h"
#include "Config/unistd.h"
#include "Config/sys/types.h"
#include "Config/sys/stat.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

static const char CacheRoot[] = "/tmp/LLVMCache";
static const char ExeSuffix[] = ".exe";

char* computeCachedFile(const char *key) {
  /* CacheRoot + "/" + std::string(key) + ExeSuffix; */
  char *cacheFile = (char*) malloc(strlen(CacheRoot) + 1 + strlen(key) + 
                                   strlen(ExeSuffix) + 1);
  char *pCacheFile = cacheFile;
  if (!cacheFile) return 0;
  memcpy(cacheFile, CacheRoot, strlen(CacheRoot));
  pCacheFile += strlen(CacheRoot);
  *pCacheFile++ = '/';
  memcpy(pCacheFile, key, strlen(key));
  pCacheFile += strlen(key);
  memcpy(pCacheFile, ExeSuffix, strlen(ExeSuffix));
  pCacheFile += strlen(ExeSuffix);
  *pCacheFile = 0; // Null-terminate
  return cacheFile;
}

/*
 * llvmStat - equivalent to stat(3), except the key may not necessarily
 * correspond to a file by that name, implementation is up to the OS.
 * Values returned in buf are similar as they are in Unix.
 */
void llvmStat(const char *key, struct stat *buf) {
  char* cacheFile = computeCachedFile(key);
  fprintf(stderr, "llvmStat(%s)\n", cacheFile);
  stat(cacheFile, buf);
  free(cacheFile);
}

/*
 * llvmWriteFile - implements a 'save' of a file in the OS. 'key' may not
 * necessarily map to a file of the same name.
 * Returns:
 * 0 - success
 * non-zero - error
 */ 
int llvmWriteFile(const char *key, const void *data, size_t len)
{
  char* cacheFile = computeCachedFile(key);
  int fd = open(cacheFile, O_CREAT|O_WRONLY|O_TRUNC);
  free(cacheFile);
  if (fd < 0) return -1; // encountered an error
  if (write(fd, data, len)) return -1;
  if (fsync(fd)) return -1;
  if (close(fd)) return -1;
  return 0;
}

/* 
 * llvmReadFile - tells the OS to load data corresponding to a particular key
 * somewhere into memory.
 * Returns:
 * 0 - failure
 * non-zero - address of loaded file
 *
 * Value of size is the length of data loaded into memory.
 */ 
void* llvmReadFile(const char *key, size_t *size) {
  char* cacheFile = computeCachedFile(key);
  if (!cacheFile) return 0;
  struct stat buf;
  stat(cacheFile, &buf);
  int fd = open(cacheFile, O_RDONLY);
  if (fd < 0) return 0; // encountered an error
  void* data = malloc(buf.st_size);
  if (read(fd, data, buf.st_size)) {
    free(data);  
    return 0;
  }
  *size = buf.st_size;
  return data;
}

/*
 * llvmExecve - execute a file from cache. This is a temporary proof-of-concept
 * because we do not relocate what we can read from disk.
 */
int llvmExecve(const char *filename, char *const argv[], char *const envp[]) {
  char* cacheFile = computeCachedFile(filename);
  executeProgram(cacheFile, argv, envp);
}
