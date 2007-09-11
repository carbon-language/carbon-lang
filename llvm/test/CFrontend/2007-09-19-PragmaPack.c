// RUN: %llvmgcc -O3 -S -o - %s | grep {26}

#include <stdint.h>

#pragma pack(push, 1)
typedef struct
{
        uint32_t        a;
        uint8_t *       b;
} foo;

typedef struct {
        uint8_t         major;
        uint8_t         minor;
        uint16_t        build;
} VERSION;

typedef struct {
        uint8_t       a[5];
        VERSION       version;
        uint8_t       b;
        char *        c;
        foo           d;
        uint32_t      guard;
} bar;
#pragma pack(pop)


unsigned barsize(void) {
  return sizeof(bar);
}

