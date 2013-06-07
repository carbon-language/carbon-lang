//===-- sanitizer_platform_limits_posix.h ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of Sanitizer common code.
//
// Sizes and layouts of platform-specific POSIX data structures.
//===----------------------------------------------------------------------===//

#ifndef SANITIZER_PLATFORM_LIMITS_POSIX_H
#define SANITIZER_PLATFORM_LIMITS_POSIX_H

#include "sanitizer_platform.h"

namespace __sanitizer {
  extern unsigned struct_utsname_sz;
  extern unsigned struct_stat_sz;
  extern unsigned struct_stat64_sz;
  extern unsigned struct_rusage_sz;
  extern unsigned struct_tm_sz;
  extern unsigned struct_passwd_sz;
  extern unsigned struct_group_sz;
  extern unsigned struct_sigaction_sz;
  extern unsigned siginfo_t_sz;
  extern unsigned struct_itimerval_sz;
  extern unsigned pthread_t_sz;
  extern unsigned pid_t_sz;
  extern unsigned timeval_sz;
  extern unsigned uid_t_sz;

#if !SANITIZER_ANDROID
  extern unsigned ucontext_t_sz;
#endif // !SANITIZER_ANDROID

#if SANITIZER_LINUX
  extern unsigned struct_rlimit_sz;
  extern unsigned struct_dirent_sz;
  extern unsigned struct_statfs_sz;
  extern unsigned struct_epoll_event_sz;
  extern unsigned struct_timespec_sz;
#endif // SANITIZER_LINUX

#if SANITIZER_LINUX && !SANITIZER_ANDROID
  extern unsigned struct_dirent64_sz;
  extern unsigned struct_rlimit64_sz;
  extern unsigned struct_statfs64_sz;
#endif // SANITIZER_LINUX && !SANITIZER_ANDROID

  struct __sanitizer_iovec {
    void  *iov_base;
    uptr iov_len;
  };

#if SANITIZER_ANDROID || SANITIZER_MAC
  struct __sanitizer_msghdr {
    void *msg_name;
    unsigned msg_namelen;
    struct __sanitizer_iovec *msg_iov;
    unsigned msg_iovlen;
    void *msg_control;
    unsigned msg_controllen;
    int msg_flags;
  };
  struct __sanitizer_cmsghdr {
    unsigned cmsg_len;
    int cmsg_level;
    int cmsg_type;
  };
#else
  struct __sanitizer_msghdr {
    void *msg_name;
    unsigned msg_namelen;
    struct __sanitizer_iovec *msg_iov;
    uptr msg_iovlen;
    void *msg_control;
    uptr msg_controllen;
    int msg_flags;
  };
  struct __sanitizer_cmsghdr {
    uptr cmsg_len;
    int cmsg_level;
    int cmsg_type;
  };
#endif

  // This thing depends on the platform. We are only interested in the upper
  // limit. Verified with a compiler assert in .cc.
  const int pthread_attr_t_max_sz = 128;
  union __sanitizer_pthread_attr_t {
    char size[pthread_attr_t_max_sz]; // NOLINT
    void *align;
  };

  uptr __sanitizer_get_sigaction_sa_sigaction(void *act);
  void __sanitizer_set_sigaction_sa_sigaction(void *act, uptr cb);
  bool __sanitizer_get_sigaction_sa_siginfo(void *act);

  const unsigned struct_sigaction_max_sz = 256;
  union __sanitizer_sigaction {
    char size[struct_sigaction_max_sz]; // NOLINT
  };

  extern uptr sig_ign;
  extern uptr sig_dfl;

#if SANITIZER_LINUX
  extern int e_tabsz;
#endif

  uptr __sanitizer_in_addr_sz(int af);

#if SANITIZER_LINUX
  struct __sanitizer_dl_phdr_info {
    uptr dlpi_addr;
    const char *dlpi_name;
    const void *dlpi_phdr;
    short dlpi_phnum;
  };
#endif

  struct __sanitizer_addrinfo {
    int ai_flags;
    int ai_family;
    int ai_socktype;
    int ai_protocol;
#if SANITIZER_ANDROID || SANITIZER_MAC
    unsigned ai_addrlen;
    char *ai_canonname;
    void *ai_addr;
#else // LINUX
    unsigned ai_addrlen;
    void *ai_addr;
    char *ai_canonname;
#endif
    struct __sanitizer_addrinfo *ai_next;
  };

  struct __sanitizer_hostent {
    char *h_name;
    char **h_aliases;
    int h_addrtype;
    int h_length;
    char **h_addr_list;
  };

  // ioctl arguments
  struct __sanitizer_ifconf {
    int ifc_len;
    union {
      void *ifcu_req;
    } ifc_ifcu;
  };

  extern unsigned struct_arpreq_sz;
  extern unsigned struct_ifreq_sz;
  extern unsigned struct_termios_sz;
  extern unsigned struct_winsize_sz;

#if SANITIZER_LINUX
  extern unsigned struct_cdrom_msf_sz;
  extern unsigned struct_cdrom_multisession_sz;
  extern unsigned struct_cdrom_read_audio_sz;
  extern unsigned struct_cdrom_subchnl_sz;
  extern unsigned struct_cdrom_ti_sz;
  extern unsigned struct_cdrom_tocentry_sz;
  extern unsigned struct_cdrom_tochdr_sz;
  extern unsigned struct_cdrom_volctrl_sz;
  extern unsigned struct_copr_buffer_sz;
  extern unsigned struct_copr_debug_buf_sz;
  extern unsigned struct_copr_msg_sz;
  extern unsigned struct_floppy_drive_params_sz;
  extern unsigned struct_floppy_drive_struct_sz;
  extern unsigned struct_floppy_fdc_state_sz;
  extern unsigned struct_floppy_max_errors_sz;
  extern unsigned struct_floppy_raw_cmd_sz;
  extern unsigned struct_floppy_struct_sz;
  extern unsigned struct_floppy_write_errors_sz;
  extern unsigned struct_format_descr_sz;
  extern unsigned struct_hd_driveid_sz;
  extern unsigned struct_hd_geometry_sz;
  extern unsigned struct_midi_info_sz;
  extern unsigned struct_mtget_sz;
  extern unsigned struct_mtop_sz;
  extern unsigned struct_mtpos_sz;
  extern unsigned struct_rtentry_sz;
  extern unsigned struct_sbi_instrument_sz;
  extern unsigned struct_seq_event_rec_sz;
  extern unsigned struct_synth_info_sz;
  extern unsigned struct_termio_sz;
  extern unsigned struct_vt_consize_sz;
  extern unsigned struct_vt_mode_sz;
  extern unsigned struct_vt_sizes_sz;
  extern unsigned struct_vt_stat_sz;
#endif

#if SANITIZER_LINUX && !SANITIZER_ANDROID
  extern unsigned struct_audio_buf_info_sz;
  extern unsigned struct_ax25_parms_struct_sz;
  extern unsigned struct_cyclades_monitor_sz;
  extern unsigned struct_ipx_config_data_sz;
  extern unsigned struct_kbdiacrs_sz;
  extern unsigned struct_kbentry_sz;
  extern unsigned struct_kbkeycode_sz;
  extern unsigned struct_kbsentry_sz;
  extern unsigned mpu_command_rec_sz;
  extern unsigned struct_mtconfiginfo_sz;
  extern unsigned struct_nr_parms_struct_sz;
  extern unsigned struct_ppp_stats_sz;
  extern unsigned struct_scc_modem_sz;
  extern unsigned struct_scc_stat_sz;
  extern unsigned struct_serial_multiport_struct_sz;
  extern unsigned struct_serial_struct_sz;
  extern unsigned struct_sockaddr_ax25_sz;
  extern unsigned struct_unimapdesc_sz;
  extern unsigned struct_unimapinit_sz;
#endif
  
#if !SANITIZER_ANDROID
  extern unsigned struct_sioc_sg_req_sz;
  extern unsigned struct_sioc_vif_req_sz;
#endif
}  // namespace __sanitizer

#endif

