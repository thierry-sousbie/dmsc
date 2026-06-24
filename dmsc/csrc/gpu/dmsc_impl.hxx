#pragma once

#ifdef __APPLE__
#include "./dmsc_mps.hxx"
#else
#include "./dmsc_cuda.hxx"
#endif

#include "./workspace.hxx"