#!/usr/bin/env python 

class BoloMassError(Exception):
    """Base class for errors in BoloMass; will move this elsewhere later"""
    pass

class BadInputError(BoloMassError):
    """Errors resulting from the user not knowing what they're doing"""
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return self.msg
