// RUN: %clangxx_msan -m64 -O0 -g %s -o %t && %t
// RUN: %clangxx_msan -m64 -O3 -g %s -o %t && %t

#include <assert.h>
#include <fcntl.h>
#include <sound/asound.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <sanitizer/msan_interface.h>

int main(int argc, char **argv) {
  int fd = open("/dev/snd/controlC0", O_RDONLY);
  if (fd < 0) {
    printf("Unable to open sound device.");
    return 0;
  }
  snd_ctl_card_info info;
  assert(__msan_test_shadow(&info, sizeof(info)) != -1);
  assert(ioctl(fd, SNDRV_CTL_IOCTL_CARD_INFO, &info) >= 0);
  assert(__msan_test_shadow(&info, sizeof(info)) == -1);
  close(fd);
  return 0;
}
