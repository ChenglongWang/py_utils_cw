#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from logging import Formatter, handlers, \
                    StreamHandler, getLogger, \
                    DEBUG, INFO, WARNING, ERROR, CRITICAL
from termcolor import COLORS
from .utils import check_dir

color_map = {CRITICAL:'magenta', ERROR:'red', WARNING:'yellow', INFO:'green', DEBUG: 'white'}

def is_color_support():
    """
    Returns True if the running system's terminal supports color, and False otherwise.
    """
    plat = os.sys.platform
    supported_platform = plat != 'Pocket PC' and (plat != 'win32' or
                                                  'ANSICON' in os.environ)
    # isatty is not always implemented, #6223.
    is_a_tty = hasattr(os.sys.stdout, 'isatty') and os.sys.stdout.isatty()
    return False if not supported_platform or not is_a_tty else True

class ColorFormatter(Formatter):
    """ A log formatter with color injection. """

    def __init__(self, fmt='[%(asctime)s][%(name)s]:  %(message)s',
                 datefmt='%H:%M:%S', reset='\x1b[0m'):
        """ Better format defaults. Reset code can be overridden if necessary."""

        Formatter.__init__(self, fmt=fmt, datefmt=datefmt)
        self.reset = reset
        self.colormap = color_map

    def format(self, record):
        """ Inject color codes & color resets into log record messages. """
        message = Formatter.format(self, record)
        fmt_str = '\x1b[%dm'
        
        try:
            color = fmt_str % (COLORS[self.colormap[record.levelno]])
            message = color + message + self.reset
        except:
            print('Error in ColorFormatter')

        return message

class Logger:
    def __init__(self, name=__name__, level='debug', output_dir=None):
        level_map = {'debug':DEBUG, 'info':INFO, 'warning':WARNING, 'error':ERROR}        
        self.logger = getLogger(name)
        self.logger.setLevel(level_map[level])
        self.color_ok = is_color_support()

        formatter = Formatter("[%(asctime)s][%(name)s][%(levelname)-8s]: %(message)s",datefmt='%Y-%m-%d %H:%M:%S')
        color_formatter = ColorFormatter(datefmt='%m-%d %H:%M')

        # stdout
        handler = StreamHandler()
        handler.setLevel(level_map[level])
        if self.color_ok:
            handler.setFormatter(color_formatter)
        else:
            handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # file
        if output_dir is not None:
            output_fname = check_dir(output_dir, name+'.log', isFile=True)
            handler = handlers.RotatingFileHandler(filename = output_fname,
                                                    maxBytes = 2e+6,
                                                    backupCount = 1)
            handler.setLevel(level_map[level])
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warn(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def critical(self, msg):
        self.logger.critical(msg)
