#!/usr/bin/env python 

import sys

class VerboseMsg(object):
    """Shortcut to write debugging messages"""
    def __init__(self, prefix="", verbose=True, flush=True):
        self.prefix, self.verbose, self.flush = prefix, verbose, flush
    def __call__(self, *args):
        if self.verbose:
            print "{0}: ".format(self.prefix),
            for a in args:  print a,
            print
            if self.flush:
                sys.stdout.flush()
