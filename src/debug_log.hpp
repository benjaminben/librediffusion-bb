#pragma once

#include <sstream>
#include <string>

#ifdef _WIN32
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <windows.h>
  inline void debug_log_line(const std::string& s)
  {
    OutputDebugStringA(("[librediffusion] " + s + "\n").c_str());
  }
#else
  #include <cstdio>
  inline void debug_log_line(const std::string& s)
  {
    fprintf(stderr, "[librediffusion] %s\n", s.c_str());
    fflush(stderr);
  }
#endif

#define DBG(expr)                                                              \
  do                                                                           \
  {                                                                            \
    std::ostringstream _dbg_oss;                                               \
    _dbg_oss << expr;                                                          \
    debug_log_line(_dbg_oss.str());                                            \
  } while(0)
