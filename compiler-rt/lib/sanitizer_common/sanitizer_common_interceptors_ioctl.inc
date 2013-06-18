//===-- sanitizer_common_interceptors_ioctl.inc -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Ioctl handling in common sanitizer interceptors.
//===----------------------------------------------------------------------===//

#include "sanitizer_flags.h"

struct ioctl_desc {
  unsigned req;
  // FIXME: support read+write arguments. Those are currently marked as WRITE.
  enum {
    NONE,
    READ,
    WRITE,
    CUSTOM
  } type : 2;
  unsigned size : 30;
  const char* name;
};

const unsigned ioctl_table_max = 500;
static ioctl_desc ioctl_table[ioctl_table_max];
static unsigned ioctl_table_size = 0;

// This can not be declared as a global, because references to struct_*_sz
// require a global initializer. And this table must be available before global
// initializers are run.
static void ioctl_table_fill() {
#define _(rq, tp, sz)                                  \
  CHECK(ioctl_table_size < ioctl_table_max);           \
  ioctl_table[ioctl_table_size].req = IOCTL_##rq;      \
  ioctl_table[ioctl_table_size].type = ioctl_desc::tp; \
  ioctl_table[ioctl_table_size].size = sz;             \
  ioctl_table[ioctl_table_size].name = #rq;            \
  ++ioctl_table_size;

  _(FIOASYNC, READ, sizeof(int));
  _(FIOCLEX, NONE, 0);
  _(FIOGETOWN, WRITE, sizeof(int));
  _(FIONBIO, READ, sizeof(int));
  _(FIONCLEX, NONE, 0);
  _(FIOSETOWN, READ, sizeof(int));
  _(SIOCADDMULTI, READ, struct_ifreq_sz);
  _(SIOCATMARK, WRITE, sizeof(int));
  _(SIOCDELMULTI, READ, struct_ifreq_sz);
  _(SIOCGIFADDR, WRITE, struct_ifreq_sz);
  _(SIOCGIFBRDADDR, WRITE, struct_ifreq_sz);
  _(SIOCGIFCONF, CUSTOM, 0);
  _(SIOCGIFDSTADDR, WRITE, struct_ifreq_sz);
  _(SIOCGIFFLAGS, WRITE, struct_ifreq_sz);
  _(SIOCGIFMETRIC, WRITE, struct_ifreq_sz);
  _(SIOCGIFMTU, WRITE, struct_ifreq_sz);
  _(SIOCGIFNETMASK, WRITE, struct_ifreq_sz);
  _(SIOCGPGRP, WRITE, sizeof(int));
  _(SIOCSIFADDR, READ, struct_ifreq_sz);
  _(SIOCSIFBRDADDR, READ, struct_ifreq_sz);
  _(SIOCSIFDSTADDR, READ, struct_ifreq_sz);
  _(SIOCSIFFLAGS, READ, struct_ifreq_sz);
  _(SIOCSIFMETRIC, READ, struct_ifreq_sz);
  _(SIOCSIFMTU, READ, struct_ifreq_sz);
  _(SIOCSIFNETMASK, READ, struct_ifreq_sz);
  _(SIOCSPGRP, READ, sizeof(int));
  _(TIOCCONS, NONE, 0);
  _(TIOCEXCL, NONE, 0);
  _(TIOCGETD, WRITE, sizeof(int));
  _(TIOCGPGRP, WRITE, pid_t_sz);
  _(TIOCGWINSZ, WRITE, struct_winsize_sz);
  _(TIOCMBIC, READ, sizeof(int));
  _(TIOCMBIS, READ, sizeof(int));
  _(TIOCMGET, WRITE, sizeof(int));
  _(TIOCMSET, READ, sizeof(int));
  _(TIOCNOTTY, NONE, 0);
  _(TIOCNXCL, NONE, 0);
  _(TIOCOUTQ, WRITE, sizeof(int));
  _(TIOCPKT, READ, sizeof(int));
  _(TIOCSCTTY, NONE, 0);
  _(TIOCSETD, READ, sizeof(int));
  _(TIOCSPGRP, READ, pid_t_sz);
  _(TIOCSTI, READ, sizeof(char));
  _(TIOCSWINSZ, READ, struct_winsize_sz);

#if (SANITIZER_LINUX && !SANITIZER_ANDROID) || SANITIZER_MAC
  _(SIOCGETSGCNT, WRITE, struct_sioc_sg_req_sz);
  _(SIOCGETVIFCNT, WRITE, struct_sioc_vif_req_sz);
#endif

#if SANITIZER_LINUX
  // Conflicting request ids.
  // _(CDROMAUDIOBUFSIZ, NONE, 0);
  // _(SNDCTL_TMR_CONTINUE, NONE, 0);
  // _(SNDCTL_TMR_START, NONE, 0);
  // _(SNDCTL_TMR_STOP, NONE, 0);
  // _(SOUND_MIXER_READ_LOUD, WRITE, sizeof(int)); // same as ...READ_ENHANCE
  // _(SOUND_MIXER_READ_MUTE, WRITE, sizeof(int)); // same as ...READ_ENHANCE
  // _(SOUND_MIXER_WRITE_LOUD, WRITE, sizeof(int)); // same as ...WRITE_ENHANCE
  // _(SOUND_MIXER_WRITE_MUTE, WRITE, sizeof(int)); // same as ...WRITE_ENHANCE
  _(BLKFLSBUF, NONE, 0);
  _(BLKGETSIZE, WRITE, sizeof(uptr));
  _(BLKRAGET, WRITE, sizeof(int));
  _(BLKRASET, NONE, 0);
  _(BLKROGET, WRITE, sizeof(int));
  _(BLKROSET, READ, sizeof(int));
  _(BLKRRPART, NONE, 0);
  _(CDROMEJECT, NONE, 0);
  _(CDROMEJECT_SW, NONE, 0);
  _(CDROMMULTISESSION, WRITE, struct_cdrom_multisession_sz);
  _(CDROMPAUSE, NONE, 0);
  _(CDROMPLAYMSF, READ, struct_cdrom_msf_sz);
  _(CDROMPLAYTRKIND, READ, struct_cdrom_ti_sz);
  _(CDROMREADAUDIO, READ, struct_cdrom_read_audio_sz);
  _(CDROMREADCOOKED, READ, struct_cdrom_msf_sz);
  _(CDROMREADMODE1, READ, struct_cdrom_msf_sz);
  _(CDROMREADMODE2, READ, struct_cdrom_msf_sz);
  _(CDROMREADRAW, READ, struct_cdrom_msf_sz);
  _(CDROMREADTOCENTRY, WRITE, struct_cdrom_tocentry_sz);
  _(CDROMREADTOCHDR, WRITE, struct_cdrom_tochdr_sz);
  _(CDROMRESET, NONE, 0);
  _(CDROMRESUME, NONE, 0);
  _(CDROMSEEK, READ, struct_cdrom_msf_sz);
  _(CDROMSTART, NONE, 0);
  _(CDROMSTOP, NONE, 0);
  _(CDROMSUBCHNL, WRITE, struct_cdrom_subchnl_sz);
  _(CDROMVOLCTRL, READ, struct_cdrom_volctrl_sz);
  _(CDROMVOLREAD, WRITE, struct_cdrom_volctrl_sz);
  _(CDROM_GET_UPC, WRITE, 8);
  _(FDCLRPRM, NONE, 0);
  _(FDDEFPRM, READ, struct_floppy_struct_sz);
  _(FDFLUSH, NONE, 0);
  _(FDFMTBEG, NONE, 0);
  _(FDFMTEND, NONE, 0);
  _(FDFMTTRK, READ, struct_format_descr_sz);
  _(FDGETDRVPRM, WRITE, struct_floppy_drive_params_sz);
  _(FDGETDRVSTAT, WRITE, struct_floppy_drive_struct_sz);
  _(FDGETDRVTYP, WRITE, 16);
  _(FDGETFDCSTAT, WRITE, struct_floppy_fdc_state_sz);
  _(FDGETMAXERRS, WRITE, struct_floppy_max_errors_sz);
  _(FDGETPRM, WRITE, struct_floppy_struct_sz);
  _(FDMSGOFF, NONE, 0);
  _(FDMSGON, NONE, 0);
  _(FDPOLLDRVSTAT, WRITE, struct_floppy_drive_struct_sz);
  _(FDRAWCMD, WRITE, struct_floppy_raw_cmd_sz);
  _(FDRESET, NONE, 0);
  _(FDSETDRVPRM, READ, struct_floppy_drive_params_sz);
  _(FDSETEMSGTRESH, NONE, 0);
  _(FDSETMAXERRS, READ, struct_floppy_max_errors_sz);
  _(FDSETPRM, READ, struct_floppy_struct_sz);
  _(FDTWADDLE, NONE, 0);
  _(FDWERRORCLR, NONE, 0);
  _(FDWERRORGET, WRITE, struct_floppy_write_errors_sz);
  _(HDIO_DRIVE_CMD, WRITE, sizeof(int));
  _(HDIO_GETGEO, WRITE, struct_hd_geometry_sz);
  _(HDIO_GET_32BIT, WRITE, sizeof(int));
  _(HDIO_GET_DMA, WRITE, sizeof(int));
  _(HDIO_GET_IDENTITY, WRITE, struct_hd_driveid_sz);
  _(HDIO_GET_KEEPSETTINGS, WRITE, sizeof(int));
  _(HDIO_GET_MULTCOUNT, WRITE, sizeof(int));
  _(HDIO_GET_NOWERR, WRITE, sizeof(int));
  _(HDIO_GET_UNMASKINTR, WRITE, sizeof(int));
  _(HDIO_SET_32BIT, NONE, 0);
  _(HDIO_SET_DMA, NONE, 0);
  _(HDIO_SET_KEEPSETTINGS, NONE, 0);
  _(HDIO_SET_MULTCOUNT, NONE, 0);
  _(HDIO_SET_NOWERR, NONE, 0);
  _(HDIO_SET_UNMASKINTR, NONE, 0);
  _(MTIOCGET, WRITE, struct_mtget_sz);
  _(MTIOCPOS, WRITE, struct_mtpos_sz);
  _(MTIOCTOP, READ, struct_mtop_sz);
  _(PPPIOCGASYNCMAP, WRITE, sizeof(int));
  _(PPPIOCGDEBUG, WRITE, sizeof(int));
  _(PPPIOCGFLAGS, WRITE, sizeof(int));
  _(PPPIOCGUNIT, WRITE, sizeof(int));
  _(PPPIOCGXASYNCMAP, WRITE, sizeof(int) * 8);
  _(PPPIOCSASYNCMAP, READ, sizeof(int));
  _(PPPIOCSDEBUG, READ, sizeof(int));
  _(PPPIOCSFLAGS, READ, sizeof(int));
  _(PPPIOCSMAXCID, READ, sizeof(int));
  _(PPPIOCSMRU, READ, sizeof(int));
  _(PPPIOCSXASYNCMAP, READ, sizeof(int) * 8);
  _(SIOCADDRT, READ, struct_rtentry_sz);
  _(SIOCDARP, READ, struct_arpreq_sz);
  _(SIOCDELRT, READ, struct_rtentry_sz);
  _(SIOCDRARP, READ, struct_arpreq_sz);
  _(SIOCGARP, WRITE, struct_arpreq_sz);
  _(SIOCGIFENCAP, WRITE, sizeof(int));
  _(SIOCGIFHWADDR, WRITE, struct_ifreq_sz);
  _(SIOCGIFMAP, WRITE, struct_ifreq_sz);
  _(SIOCGIFMEM, WRITE, struct_ifreq_sz);
  _(SIOCGIFNAME, NONE, 0);
  _(SIOCGIFSLAVE, NONE, 0);
  _(SIOCGRARP, WRITE, struct_arpreq_sz);
  _(SIOCGSTAMP, WRITE, timeval_sz);
  _(SIOCSARP, READ, struct_arpreq_sz);
  _(SIOCSIFENCAP, READ, sizeof(int));
  _(SIOCSIFHWADDR, READ, struct_ifreq_sz);
  _(SIOCSIFLINK, NONE, 0);
  _(SIOCSIFMAP, READ, struct_ifreq_sz);
  _(SIOCSIFMEM, READ, struct_ifreq_sz);
  _(SIOCSIFSLAVE, NONE, 0);
  _(SIOCSRARP, READ, struct_arpreq_sz);
  _(SNDCTL_COPR_HALT, WRITE, struct_copr_debug_buf_sz);
  _(SNDCTL_COPR_LOAD, READ, struct_copr_buffer_sz);
  _(SNDCTL_COPR_RCODE, WRITE, struct_copr_debug_buf_sz);
  _(SNDCTL_COPR_RCVMSG, WRITE, struct_copr_msg_sz);
  _(SNDCTL_COPR_RDATA, WRITE, struct_copr_debug_buf_sz);
  _(SNDCTL_COPR_RESET, NONE, 0);
  _(SNDCTL_COPR_RUN, WRITE, struct_copr_debug_buf_sz);
  _(SNDCTL_COPR_SENDMSG, READ, struct_copr_msg_sz);
  _(SNDCTL_COPR_WCODE, READ, struct_copr_debug_buf_sz);
  _(SNDCTL_COPR_WDATA, READ, struct_copr_debug_buf_sz);
  _(SNDCTL_DSP_GETBLKSIZE, WRITE, sizeof(int));
  _(SNDCTL_DSP_GETFMTS, WRITE, sizeof(int));
  _(SNDCTL_DSP_NONBLOCK, NONE, 0);
  _(SNDCTL_DSP_POST, NONE, 0);
  _(SNDCTL_DSP_RESET, NONE, 0);
  _(SNDCTL_DSP_SETFMT, WRITE, sizeof(int));
  _(SNDCTL_DSP_SETFRAGMENT, WRITE, sizeof(int));
  _(SNDCTL_DSP_SPEED, WRITE, sizeof(int));
  _(SNDCTL_DSP_STEREO, WRITE, sizeof(int));
  _(SNDCTL_DSP_SUBDIVIDE, WRITE, sizeof(int));
  _(SNDCTL_DSP_SYNC, NONE, 0);
  _(SNDCTL_FM_4OP_ENABLE, READ, sizeof(int));
  _(SNDCTL_FM_LOAD_INSTR, READ, struct_sbi_instrument_sz);
  _(SNDCTL_MIDI_INFO, WRITE, struct_midi_info_sz);
  _(SNDCTL_MIDI_MPUMODE, READ, sizeof(int));
  _(SNDCTL_MIDI_PRETIME, WRITE, sizeof(int));
  _(SNDCTL_SEQ_CTRLRATE, WRITE, sizeof(int));
  _(SNDCTL_SEQ_GETINCOUNT, WRITE, sizeof(int));
  _(SNDCTL_SEQ_GETOUTCOUNT, WRITE, sizeof(int));
  _(SNDCTL_SEQ_NRMIDIS, WRITE, sizeof(int));
  _(SNDCTL_SEQ_NRSYNTHS, WRITE, sizeof(int));
  _(SNDCTL_SEQ_OUTOFBAND, READ, struct_seq_event_rec_sz);
  _(SNDCTL_SEQ_PANIC, NONE, 0);
  _(SNDCTL_SEQ_PERCMODE, NONE, 0);
  _(SNDCTL_SEQ_RESET, NONE, 0);
  _(SNDCTL_SEQ_RESETSAMPLES, READ, sizeof(int));
  _(SNDCTL_SEQ_SYNC, NONE, 0);
  _(SNDCTL_SEQ_TESTMIDI, READ, sizeof(int));
  _(SNDCTL_SEQ_THRESHOLD, READ, sizeof(int));
  _(SNDCTL_SYNTH_INFO, WRITE, struct_synth_info_sz);
  _(SNDCTL_SYNTH_MEMAVL, WRITE, sizeof(int));
  _(SNDCTL_TMR_METRONOME, READ, sizeof(int));
  _(SNDCTL_TMR_SELECT, WRITE, sizeof(int));
  _(SNDCTL_TMR_SOURCE, WRITE, sizeof(int));
  _(SNDCTL_TMR_TEMPO, WRITE, sizeof(int));
  _(SNDCTL_TMR_TIMEBASE, WRITE, sizeof(int));
  _(SOUND_MIXER_READ_ALTPCM, WRITE, sizeof(int));
  _(SOUND_MIXER_READ_BASS, WRITE, sizeof(int));
  _(SOUND_MIXER_READ_CAPS, WRITE, sizeof(int));
  _(SOUND_MIXER_READ_CD, WRITE, sizeof(int));
  _(SOUND_MIXER_READ_DEVMASK, WRITE, sizeof(int));
  _(SOUND_MIXER_READ_ENHANCE, WRITE, sizeof(int));
  _(SOUND_MIXER_READ_IGAIN, WRITE, sizeof(int));
  _(SOUND_MIXER_READ_IMIX, WRITE, sizeof(int));
  _(SOUND_MIXER_READ_LINE, WRITE, sizeof(int));
  _(SOUND_MIXER_READ_LINE1, WRITE, sizeof(int));
  _(SOUND_MIXER_READ_LINE2, WRITE, sizeof(int));
  _(SOUND_MIXER_READ_LINE3, WRITE, sizeof(int));
  _(SOUND_MIXER_READ_MIC, WRITE, sizeof(int));
  _(SOUND_MIXER_READ_OGAIN, WRITE, sizeof(int));
  _(SOUND_MIXER_READ_PCM, WRITE, sizeof(int));
  _(SOUND_MIXER_READ_RECLEV, WRITE, sizeof(int));
  _(SOUND_MIXER_READ_RECMASK, WRITE, sizeof(int));
  _(SOUND_MIXER_READ_RECSRC, WRITE, sizeof(int));
  _(SOUND_MIXER_READ_SPEAKER, WRITE, sizeof(int));
  _(SOUND_MIXER_READ_STEREODEVS, WRITE, sizeof(int));
  _(SOUND_MIXER_READ_SYNTH, WRITE, sizeof(int));
  _(SOUND_MIXER_READ_TREBLE, WRITE, sizeof(int));
  _(SOUND_MIXER_READ_VOLUME, WRITE, sizeof(int));
  _(SOUND_MIXER_WRITE_ALTPCM, WRITE, sizeof(int));
  _(SOUND_MIXER_WRITE_BASS, WRITE, sizeof(int));
  _(SOUND_MIXER_WRITE_CD, WRITE, sizeof(int));
  _(SOUND_MIXER_WRITE_ENHANCE, WRITE, sizeof(int));
  _(SOUND_MIXER_WRITE_IGAIN, WRITE, sizeof(int));
  _(SOUND_MIXER_WRITE_IMIX, WRITE, sizeof(int));
  _(SOUND_MIXER_WRITE_LINE, WRITE, sizeof(int));
  _(SOUND_MIXER_WRITE_LINE1, WRITE, sizeof(int));
  _(SOUND_MIXER_WRITE_LINE2, WRITE, sizeof(int));
  _(SOUND_MIXER_WRITE_LINE3, WRITE, sizeof(int));
  _(SOUND_MIXER_WRITE_MIC, WRITE, sizeof(int));
  _(SOUND_MIXER_WRITE_OGAIN, WRITE, sizeof(int));
  _(SOUND_MIXER_WRITE_PCM, WRITE, sizeof(int));
  _(SOUND_MIXER_WRITE_RECLEV, WRITE, sizeof(int));
  _(SOUND_MIXER_WRITE_RECSRC, WRITE, sizeof(int));
  _(SOUND_MIXER_WRITE_SPEAKER, WRITE, sizeof(int));
  _(SOUND_MIXER_WRITE_SYNTH, WRITE, sizeof(int));
  _(SOUND_MIXER_WRITE_TREBLE, WRITE, sizeof(int));
  _(SOUND_MIXER_WRITE_VOLUME, WRITE, sizeof(int));
  _(SOUND_PCM_READ_BITS, WRITE, sizeof(int));
  _(SOUND_PCM_READ_CHANNELS, WRITE, sizeof(int));
  _(SOUND_PCM_READ_FILTER, WRITE, sizeof(int));
  _(SOUND_PCM_READ_RATE, WRITE, sizeof(int));
  _(SOUND_PCM_WRITE_CHANNELS, WRITE, sizeof(int));
  _(SOUND_PCM_WRITE_FILTER, WRITE, sizeof(int));
  _(TCFLSH, NONE, 0);
  _(TCGETA, WRITE, struct_termio_sz);
  _(TCGETS, WRITE, struct_termios_sz);
  _(TCSBRK, NONE, 0);
  _(TCSBRKP, NONE, 0);
  _(TCSETA, READ, struct_termio_sz);
  _(TCSETAF, READ, struct_termio_sz);
  _(TCSETAW, READ, struct_termio_sz);
  _(TCSETS, READ, struct_termios_sz);
  _(TCSETSF, READ, struct_termios_sz);
  _(TCSETSW, READ, struct_termios_sz);
  _(TCXONC, NONE, 0);
  _(TIOCGLCKTRMIOS, WRITE, struct_termios_sz);
  _(TIOCGSOFTCAR, WRITE, sizeof(int));
  _(TIOCINQ, WRITE, sizeof(int));
  _(TIOCLINUX, READ, sizeof(char));
  _(TIOCSERCONFIG, NONE, 0);
  _(TIOCSERGETLSR, WRITE, sizeof(int));
  _(TIOCSERGWILD, WRITE, sizeof(int));
  _(TIOCSERSWILD, READ, sizeof(int));
  _(TIOCSLCKTRMIOS, READ, struct_termios_sz);
  _(TIOCSSOFTCAR, READ, sizeof(int));
  _(VT_ACTIVATE, NONE, 0);
  _(VT_DISALLOCATE, NONE, 0);
  _(VT_GETMODE, WRITE, struct_vt_mode_sz);
  _(VT_GETSTATE, WRITE, struct_vt_stat_sz);
  _(VT_OPENQRY, WRITE, sizeof(int));
  _(VT_RELDISP, NONE, 0);
  _(VT_RESIZE, READ, struct_vt_sizes_sz);
  _(VT_RESIZEX, READ, struct_vt_consize_sz);
  _(VT_SENDSIG, NONE, 0);
  _(VT_SETMODE, READ, struct_vt_mode_sz);
  _(VT_WAITACTIVE, NONE, 0);
#endif

#if SANITIZER_LINUX && !SANITIZER_ANDROID
  // _(SIOCDEVPLIP, WRITE, struct_ifreq_sz); // the same as EQL_ENSLAVE
  _(CYGETDEFTHRESH, WRITE, sizeof(int));
  _(CYGETDEFTIMEOUT, WRITE, sizeof(int));
  _(CYGETMON, WRITE, struct_cyclades_monitor_sz);
  _(CYGETTHRESH, WRITE, sizeof(int));
  _(CYGETTIMEOUT, WRITE, sizeof(int));
  _(CYSETDEFTHRESH, NONE, 0);
  _(CYSETDEFTIMEOUT, NONE, 0);
  _(CYSETTHRESH, NONE, 0);
  _(CYSETTIMEOUT, NONE, 0);
  _(EQL_EMANCIPATE, WRITE, struct_ifreq_sz);
  _(EQL_ENSLAVE, WRITE, struct_ifreq_sz);
  _(EQL_GETMASTRCFG, WRITE, struct_ifreq_sz);
  _(EQL_GETSLAVECFG, WRITE, struct_ifreq_sz);
  _(EQL_SETMASTRCFG, WRITE, struct_ifreq_sz);
  _(EQL_SETSLAVECFG, WRITE, struct_ifreq_sz);
  _(FS_IOC_GETFLAGS, WRITE, sizeof(int));
  _(FS_IOC_GETVERSION, WRITE, sizeof(int));
  _(FS_IOC_SETFLAGS, READ, sizeof(int));
  _(FS_IOC_SETVERSION, READ, sizeof(int));
  _(GIO_CMAP, WRITE, 48);
  _(GIO_FONT, WRITE, 8192);
  _(GIO_SCRNMAP, WRITE, e_tabsz);
  _(GIO_UNIMAP, WRITE, struct_unimapdesc_sz);
  _(GIO_UNISCRNMAP, WRITE, sizeof(short) * e_tabsz);
  _(KDADDIO, NONE, 0);
  _(KDDELIO, NONE, 0);
  _(KDDISABIO, NONE, 0);
  _(KDENABIO, NONE, 0);
  _(KDGETKEYCODE, WRITE, struct_kbkeycode_sz);
  _(KDGETLED, WRITE, 1);
  _(KDGETMODE, WRITE, sizeof(int));
  _(KDGKBDIACR, WRITE, struct_kbdiacrs_sz);
  _(KDGKBENT, WRITE, struct_kbentry_sz);
  _(KDGKBLED, WRITE, sizeof(int));
  _(KDGKBMETA, WRITE, sizeof(int));
  _(KDGKBMODE, WRITE, sizeof(int));
  _(KDGKBSENT, WRITE, struct_kbsentry_sz);
  _(KDGKBTYPE, WRITE, 1);
  _(KDMAPDISP, NONE, 0);
  _(KDMKTONE, NONE, 0);
  _(KDSETKEYCODE, READ, struct_kbkeycode_sz);
  _(KDSETLED, NONE, 0);
  _(KDSETMODE, NONE, 0);
  _(KDSIGACCEPT, NONE, 0);
  _(KDSKBDIACR, READ, struct_kbdiacrs_sz);
  _(KDSKBENT, READ, struct_kbentry_sz);
  _(KDSKBLED, NONE, 0);
  _(KDSKBMETA, NONE, 0);
  _(KDSKBMODE, NONE, 0);
  _(KDSKBSENT, READ, struct_kbsentry_sz);
  _(KDUNMAPDISP, NONE, 0);
  _(KIOCSOUND, NONE, 0);
  _(LPABORT, NONE, 0);
  _(LPABORTOPEN, NONE, 0);
  _(LPCAREFUL, NONE, 0);
  _(LPCHAR, NONE, 0);
  _(LPGETIRQ, WRITE, sizeof(int));
  _(LPGETSTATUS, WRITE, sizeof(int));
  _(LPRESET, NONE, 0);
  _(LPSETIRQ, NONE, 0);
  _(LPTIME, NONE, 0);
  _(LPWAIT, NONE, 0);
  _(MTIOCGETCONFIG, WRITE, struct_mtconfiginfo_sz);
  _(MTIOCSETCONFIG, READ, struct_mtconfiginfo_sz);
  _(PIO_CMAP, NONE, 0);
  _(PIO_FONT, READ, 8192);
  _(PIO_SCRNMAP, READ, e_tabsz);
  _(PIO_UNIMAP, READ, struct_unimapdesc_sz);
  _(PIO_UNIMAPCLR, READ, struct_unimapinit_sz);
  _(PIO_UNISCRNMAP, READ, sizeof(short) * e_tabsz);
  _(SCSI_IOCTL_PROBE_HOST, READ, sizeof(int));
  _(SCSI_IOCTL_TAGGED_DISABLE, NONE, 0);
  _(SCSI_IOCTL_TAGGED_ENABLE, NONE, 0);
  _(SNDCTL_DSP_GETISPACE, WRITE, struct_audio_buf_info_sz);
  _(SNDCTL_DSP_GETOSPACE, WRITE, struct_audio_buf_info_sz);
  _(TIOCGSERIAL, WRITE, struct_serial_struct_sz);
  _(TIOCSERGETMULTI, WRITE, struct_serial_multiport_struct_sz);
  _(TIOCSERSETMULTI, READ, struct_serial_multiport_struct_sz);
  _(TIOCSSERIAL, READ, struct_serial_struct_sz);

  // The following ioctl requests are shared between AX25, IPX, netrom and
  // mrouted.
  // _(SIOCAIPXITFCRT, READ, sizeof(char));
  // _(SIOCAX25GETUID, READ, struct_sockaddr_ax25_sz);
  // _(SIOCNRGETPARMS, WRITE, struct_nr_parms_struct_sz);
  // _(SIOCAIPXPRISLT, READ, sizeof(char));
  // _(SIOCNRSETPARMS, READ, struct_nr_parms_struct_sz);
  // _(SIOCAX25ADDUID, READ, struct_sockaddr_ax25_sz);
  // _(SIOCNRDECOBS, NONE, 0);
  // _(SIOCAX25DELUID, READ, struct_sockaddr_ax25_sz);
  // _(SIOCIPXCFGDATA, WRITE, struct_ipx_config_data_sz);
  // _(SIOCAX25NOUID, READ, sizeof(int));
  // _(SIOCNRRTCTL, READ, sizeof(int));
  // _(SIOCAX25DIGCTL, READ, sizeof(int));
  // _(SIOCAX25GETPARMS, WRITE, struct_ax25_parms_struct_sz);
  // _(SIOCAX25SETPARMS, READ, struct_ax25_parms_struct_sz);
#endif
#undef _
  // FIXME: missing USB ioctls (EVIOxx...x).
}

static bool ioctl_initialized = false;

struct ioctl_desc_compare {
  bool operator()(const ioctl_desc& left, const ioctl_desc& right) const {
    return left.req < right.req;
  }
};

static void ioctl_init() {
  ioctl_table_fill();
  InternalSort(&ioctl_table, ioctl_table_size, ioctl_desc_compare());

  bool bad = false;
  for (unsigned i = 0; i < ioctl_table_size - 1; ++i) {
    if (ioctl_table[i].req >= ioctl_table[i + 1].req) {
      Printf("Duplicate or unsorted ioctl request id %x >= %x (%s vs %s)\n",
             ioctl_table[i].req, ioctl_table[i + 1].req, ioctl_table[i].name,
             ioctl_table[i + 1].name);
      bad = true;
    }
  }

  if (bad) Die();

  ioctl_initialized = true;
}

static const ioctl_desc *ioctl_lookup(unsigned req) {
  int left = 0;
  int right = ioctl_table_size;
  while (left < right) {
    int mid = (left + right) / 2;
    if (ioctl_table[mid].req < req)
      left = mid + 1;
    else
      right = mid;
  }
  if (left == right && ioctl_table[left].req == req)
    return ioctl_table + left;
  else
    return 0;
}

static void ioctl_common_pre(void *ctx, const ioctl_desc *desc, int d,
                             unsigned request, void *arg) {
  if (desc->type == ioctl_desc::READ)
    COMMON_INTERCEPTOR_READ_RANGE(ctx, arg, desc->size);
  if (desc->type != ioctl_desc::CUSTOM)
    return;
  switch (request) {
    case 0x00008912: {  // SIOCGIFCONF
      struct __sanitizer_ifconf *ifc = (__sanitizer_ifconf *)arg;
      COMMON_INTERCEPTOR_READ_RANGE(ctx, &ifc->ifc_len, sizeof(ifc->ifc_len));
      break;
    }
  }
  return;
}

static void ioctl_common_post(void *ctx, const ioctl_desc *desc, int res, int d,
                              unsigned request, void *arg) {
  if (desc->type == ioctl_desc::WRITE) {
    // FIXME: add verbose output
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, arg, desc->size);
  }
  if (desc->type != ioctl_desc::CUSTOM)
    return;
  switch (request) {
    case 0x00008912: {  // SIOCGIFCONF
      struct __sanitizer_ifconf *ifc = (__sanitizer_ifconf *)arg;
      COMMON_INTERCEPTOR_WRITE_RANGE(ctx, ifc->ifc_ifcu.ifcu_req, ifc->ifc_len);
      break;
    }
  }
  return;
}
