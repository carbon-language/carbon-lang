//===-- sanitizer_platform_limits_posix.cc --------------------------------===//
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


#include "sanitizer_platform.h"
#if SANITIZER_LINUX || SANITIZER_MAC

#include "sanitizer_internal_defs.h"
#include "sanitizer_platform_limits_posix.h"

#include <arpa/inet.h>
#include <dirent.h>
#include <grp.h>
#include <net/if.h>
#include <net/if_arp.h>
#include <net/route.h>
#include <netdb.h>
#include <pthread.h>
#include <pwd.h>
#include <signal.h>
#include <stddef.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/socket.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/utsname.h>
#include <termios.h>
#include <time.h>

#if SANITIZER_LINUX
#include <sys/vt.h>
#include <linux/cdrom.h>
#include <linux/fd.h>
#include <linux/hdreg.h>
#include <linux/soundcard.h>
#endif

#if !SANITIZER_ANDROID
#include <sys/ucontext.h>
#endif

#if SANITIZER_LINUX && !SANITIZER_ANDROID
#include <net/if_ppp.h>
#include <netax25/ax25.h>
#include <netipx/ipx.h>
#include <netrom/netrom.h>
#include <sys/mtio.h>
#include <sys/kd.h>
#include <linux/cyclades.h>
#include <linux/lp.h>
#include <linux/mroute.h>
#include <linux/mroute6.h>
#include <linux/scc.h>
#include <linux/serial.h>
#endif // SANITIZER_LINUX && !SANITIZER_ANDROID

#if SANITIZER_ANDROID
#include <linux/kd.h>
#include <linux/mtio.h>
#endif

#if SANITIZER_LINUX
#include <link.h>
#include <sys/vfs.h>
#include <sys/epoll.h>
#endif // SANITIZER_LINUX

#if SANITIZER_MAC
#include <netinet/ip_mroute.h>
#endif

namespace __sanitizer {
  unsigned struct_utsname_sz = sizeof(struct utsname);
  unsigned struct_stat_sz = sizeof(struct stat);
  unsigned struct_stat64_sz = sizeof(struct stat64);
  unsigned struct_rusage_sz = sizeof(struct rusage);
  unsigned struct_tm_sz = sizeof(struct tm);
  unsigned struct_passwd_sz = sizeof(struct passwd);
  unsigned struct_group_sz = sizeof(struct group);
  unsigned siginfo_t_sz = sizeof(siginfo_t);
  unsigned struct_sigaction_sz = sizeof(struct sigaction);
  unsigned struct_itimerval_sz = sizeof(struct itimerval);
  unsigned pthread_t_sz = sizeof(pthread_t);
  unsigned pid_t_sz = sizeof(pid_t);
  unsigned timeval_sz = sizeof(timeval);
  unsigned uid_t_sz = sizeof(uid_t);

#if !SANITIZER_ANDROID
  unsigned ucontext_t_sz = sizeof(ucontext_t);
#endif // !SANITIZER_ANDROID

#if SANITIZER_LINUX
  unsigned struct_rlimit_sz = sizeof(struct rlimit);
  unsigned struct_dirent_sz = sizeof(struct dirent);
  unsigned struct_statfs_sz = sizeof(struct statfs);
  unsigned struct_epoll_event_sz = sizeof(struct epoll_event);
  unsigned struct_timespec_sz = sizeof(struct timespec);
#endif // SANITIZER_LINUX

#if SANITIZER_LINUX && !SANITIZER_ANDROID
  unsigned struct_dirent64_sz = sizeof(struct dirent64);
  unsigned struct_rlimit64_sz = sizeof(struct rlimit64);
  unsigned struct_statfs64_sz = sizeof(struct statfs64);
#endif // SANITIZER_LINUX && !SANITIZER_ANDROID

  uptr sig_ign = (uptr)SIG_IGN;
  uptr sig_dfl = (uptr)SIG_DFL;

#if SANITIZER_LINUX
  int e_tabsz = (int)E_TABSZ;
#endif

  uptr __sanitizer_get_sigaction_sa_sigaction(void *act) {
    struct sigaction *a = (struct sigaction *)act;
    // Check that sa_sigaction and sa_handler are the same.
    CHECK((void *)&(a->sa_sigaction) == (void *)&(a->sa_handler));
    return (uptr) a->sa_sigaction;
  }
  void __sanitizer_set_sigaction_sa_sigaction(void *act, uptr cb) {
    struct sigaction *a = (struct sigaction *)act;
    a->sa_sigaction = (void (*)(int, siginfo_t *, void *))cb;
  }
  bool __sanitizer_get_sigaction_sa_siginfo(void *act) {
    struct sigaction *a = (struct sigaction *)act;
    return a->sa_flags & SA_SIGINFO;
  }

  uptr __sanitizer_in_addr_sz(int af) {
    if (af == AF_INET)
      return sizeof(struct in_addr);
    else if (af == AF_INET6)
      return sizeof(struct in6_addr);
    else
      return 0;
  }

  // ioctl arguments
  unsigned struct_arpreq_sz = sizeof(struct arpreq);
  unsigned struct_ifconf_sz = sizeof(struct ifconf);
  unsigned struct_ifreq_sz = sizeof(struct ifreq);
  unsigned struct_termios_sz = sizeof(struct termios);
  unsigned struct_winsize_sz = sizeof(struct winsize);

#if SANITIZER_LINUX
  unsigned struct_cdrom_msf_sz = sizeof(struct cdrom_msf);
  unsigned struct_cdrom_multisession_sz = sizeof(struct cdrom_multisession);
  unsigned struct_cdrom_read_audio_sz = sizeof(struct cdrom_read_audio);
  unsigned struct_cdrom_subchnl_sz = sizeof(struct cdrom_subchnl);
  unsigned struct_cdrom_ti_sz = sizeof(struct cdrom_ti);
  unsigned struct_cdrom_tocentry_sz = sizeof(struct cdrom_tocentry);
  unsigned struct_cdrom_tochdr_sz = sizeof(struct cdrom_tochdr);
  unsigned struct_cdrom_volctrl_sz = sizeof(struct cdrom_volctrl);
  unsigned struct_copr_buffer_sz = sizeof(struct copr_buffer);
  unsigned struct_copr_debug_buf_sz = sizeof(struct copr_debug_buf);
  unsigned struct_copr_msg_sz = sizeof(struct copr_msg);
  unsigned struct_floppy_drive_params_sz = sizeof(struct floppy_drive_params);
  unsigned struct_floppy_drive_struct_sz = sizeof(struct floppy_drive_struct);
  unsigned struct_floppy_fdc_state_sz = sizeof(struct floppy_fdc_state);
  unsigned struct_floppy_max_errors_sz = sizeof(struct floppy_max_errors);
  unsigned struct_floppy_raw_cmd_sz = sizeof(struct floppy_raw_cmd);
  unsigned struct_floppy_struct_sz = sizeof(struct floppy_struct);
  unsigned struct_floppy_write_errors_sz = sizeof(struct floppy_write_errors);
  unsigned struct_format_descr_sz = sizeof(struct format_descr);
  unsigned struct_hd_driveid_sz = sizeof(struct hd_driveid);
  unsigned struct_hd_geometry_sz = sizeof(struct hd_geometry);
  unsigned struct_midi_info_sz = sizeof(struct midi_info);
  unsigned struct_mtget_sz = sizeof(struct mtget);
  unsigned struct_mtop_sz = sizeof(struct mtop);
  unsigned struct_mtpos_sz = sizeof(struct mtpos);
  unsigned struct_rtentry_sz = sizeof(struct rtentry);
  unsigned struct_sbi_instrument_sz = sizeof(struct sbi_instrument);
  unsigned struct_seq_event_rec_sz = sizeof(struct seq_event_rec);
  unsigned struct_synth_info_sz = sizeof(struct synth_info);
  unsigned struct_termio_sz = sizeof(struct termio);
  unsigned struct_vt_consize_sz = sizeof(struct vt_consize);
  unsigned struct_vt_mode_sz = sizeof(struct vt_mode);
  unsigned struct_vt_sizes_sz = sizeof(struct vt_sizes);
  unsigned struct_vt_stat_sz = sizeof(struct vt_stat);
#endif

#if SANITIZER_LINUX && !SANITIZER_ANDROID
  unsigned struct_audio_buf_info_sz = sizeof(struct audio_buf_info);
  unsigned struct_ax25_parms_struct_sz = sizeof(struct ax25_parms_struct);
  unsigned struct_cyclades_monitor_sz = sizeof(struct cyclades_monitor);
  unsigned struct_ipx_config_data_sz = sizeof(struct ipx_config_data);
  unsigned struct_kbdiacrs_sz = sizeof(struct kbdiacrs);
  unsigned struct_kbentry_sz = sizeof(struct kbentry);
  unsigned struct_kbkeycode_sz = sizeof(struct kbkeycode);
  unsigned struct_kbsentry_sz = sizeof(struct kbsentry);
  unsigned mpu_command_rec_sz = sizeof(mpu_command_rec);
  unsigned struct_mtconfiginfo_sz = sizeof(struct mtconfiginfo);
  unsigned struct_nr_parms_struct_sz = sizeof(struct nr_parms_struct);
  unsigned struct_ppp_stats_sz = sizeof(struct ppp_stats);
  unsigned struct_scc_modem_sz = sizeof(struct scc_modem);
  unsigned struct_scc_stat_sz = sizeof(struct scc_stat);
  unsigned struct_serial_multiport_struct_sz = sizeof(struct serial_multiport_struct);
  unsigned struct_serial_struct_sz = sizeof(struct serial_struct);
  unsigned struct_sockaddr_ax25_sz = sizeof(struct sockaddr_ax25);
  unsigned struct_unimapdesc_sz = sizeof(struct unimapdesc);
  unsigned struct_unimapinit_sz = sizeof(struct unimapinit);
#endif

#if !SANITIZER_ANDROID  
  unsigned struct_sioc_sg_req_sz = sizeof(struct sioc_sg_req);
  unsigned struct_sioc_vif_req_sz = sizeof(struct sioc_vif_req);
#endif
}  // namespace __sanitizer

#define CHECK_TYPE_SIZE(TYPE) \
  COMPILER_CHECK(sizeof(__sanitizer_##TYPE) == sizeof(TYPE))

#define CHECK_SIZE_AND_OFFSET(CLASS, MEMBER)                       \
  COMPILER_CHECK(sizeof(((__sanitizer_##CLASS *) NULL)->MEMBER) == \
                 sizeof(((CLASS *) NULL)->MEMBER));                \
  COMPILER_CHECK(offsetof(__sanitizer_##CLASS, MEMBER) ==          \
                 offsetof(CLASS, MEMBER))

COMPILER_CHECK(sizeof(__sanitizer_pthread_attr_t) >= sizeof(pthread_attr_t));
COMPILER_CHECK(sizeof(__sanitizer::struct_sigaction_max_sz) >=
                   sizeof(__sanitizer::struct_sigaction_sz));

COMPILER_CHECK(sizeof(socklen_t) == sizeof(unsigned));

#if SANITIZER_LINUX
// There are more undocumented fields in dl_phdr_info that we are not interested
// in.
COMPILER_CHECK(sizeof(__sanitizer_dl_phdr_info) <= sizeof(dl_phdr_info));
CHECK_SIZE_AND_OFFSET(dl_phdr_info, dlpi_addr);
CHECK_SIZE_AND_OFFSET(dl_phdr_info, dlpi_name);
CHECK_SIZE_AND_OFFSET(dl_phdr_info, dlpi_phdr);
CHECK_SIZE_AND_OFFSET(dl_phdr_info, dlpi_phnum);
#endif

CHECK_TYPE_SIZE(addrinfo);
CHECK_SIZE_AND_OFFSET(addrinfo, ai_flags);
CHECK_SIZE_AND_OFFSET(addrinfo, ai_family);
CHECK_SIZE_AND_OFFSET(addrinfo, ai_socktype);
CHECK_SIZE_AND_OFFSET(addrinfo, ai_protocol);
CHECK_SIZE_AND_OFFSET(addrinfo, ai_protocol);
CHECK_SIZE_AND_OFFSET(addrinfo, ai_addrlen);
CHECK_SIZE_AND_OFFSET(addrinfo, ai_canonname);
CHECK_SIZE_AND_OFFSET(addrinfo, ai_addr);

CHECK_TYPE_SIZE(hostent);
CHECK_SIZE_AND_OFFSET(hostent, h_name);
CHECK_SIZE_AND_OFFSET(hostent, h_aliases);
CHECK_SIZE_AND_OFFSET(hostent, h_addrtype);
CHECK_SIZE_AND_OFFSET(hostent, h_length);
CHECK_SIZE_AND_OFFSET(hostent, h_addr_list);

CHECK_TYPE_SIZE(iovec);
CHECK_SIZE_AND_OFFSET(iovec, iov_base);
CHECK_SIZE_AND_OFFSET(iovec, iov_len);

CHECK_TYPE_SIZE(msghdr);
CHECK_SIZE_AND_OFFSET(msghdr, msg_name);
CHECK_SIZE_AND_OFFSET(msghdr, msg_namelen);
CHECK_SIZE_AND_OFFSET(msghdr, msg_iov);
CHECK_SIZE_AND_OFFSET(msghdr, msg_iovlen);
CHECK_SIZE_AND_OFFSET(msghdr, msg_control);
CHECK_SIZE_AND_OFFSET(msghdr, msg_controllen);
CHECK_SIZE_AND_OFFSET(msghdr, msg_flags);

CHECK_TYPE_SIZE(cmsghdr);
CHECK_SIZE_AND_OFFSET(cmsghdr, cmsg_len);
CHECK_SIZE_AND_OFFSET(cmsghdr, cmsg_level);
CHECK_SIZE_AND_OFFSET(cmsghdr, cmsg_type);

#endif  // SANITIZER_LINUX || SANITIZER_MAC
