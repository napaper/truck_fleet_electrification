#!/usr/bin/env python
""" A wrapper for the configparser class with integrated checks
"""

# Imports: 
import configparser
import os
import warnings
from configparser import ExtendedInterpolation
import pathlib
import psycopg2


class ConfigReader:

    conf = None

    def __init__(self):
        pass

    def readConfig(self, path, config_template_path=None):
        self.conf = self.__readConfigWithoutChecking(path)
        return self.conf, self.validateConfig(config_template_path)

    # TODO: Add reporting of missing config options
    def validateConfig(self, config_template_path):

        if config_template_path is None:
            return None
        else:
            template_config = self.__readConfigWithoutChecking(config_template_path)

        if self.conf.sections()!=template_config.sections():
            return False
        else:
            for section in template_config.sections():
                config_section = self.conf[section]
                template_config_section = template_config[section]
                if config_section.keys()!=template_config_section.keys():
                    return False

        return True

    def __readConfigWithoutChecking(self, path):
        conf = configparser.ConfigParser(interpolation = ExtendedInterpolation())
        conf.read(path)
        return conf

def readConfig(path=None, config_template_path=None, verbose=False):

    if path and config_template_path:
        print("Loading config from {}".format(path))
        pass
    else:
        # Try with default paths
        cur_path = pathlib.Path(__file__).parent.resolve()
        path = os.path.abspath("{}/db.conf".format(cur_path))
        config_template_path = os.path.abspath("{}/db_template.conf".format(cur_path))
        warnings.warn("Using default path {} and template path {}".format(path, config_template_path))

    # Read config
    cfg = ConfigReader()
    config, valid = cfg.readConfig(path, config_template_path=config_template_path)

    # Validate config
    if valid is not None and not valid:
        raise Exception(
            "Config file is not valid. Please compare your file with the template provided under config/db_setup_config_template.conf. Aborting.")
    elif valid is None:
        warnings.warn("Config file has not been checked against a template file. Continuing anyways.")
    elif verbose: 
        print("The supplied config is valid. The following options where given: ")
        for section in config.sections():
            for key in config[section]:
                print("{} = {}".format(key, config[section][key]))
    else: 
        pass

    return config, valid

class DBConfig:
    
    host = None
    port = None
    name = None
    user = None
    password = None
    schema_name = None
    psql_cmd_path = None

    conf = None

    def __init__(self, path, config_template_path=None) -> None:
        self.load(path, config_template_path)
        pass

    def load(self, path, config_template_path=None):
        
        confreader = ConfigReader()
        conf, valid = confreader.readConfig(path, config_template_path)

        self.host = conf["DB"]["db_host"]
        self.port = conf["DB"]["db_port"]
        self.name = conf["DB"]["db_name"]
        self.user = conf["DB"]["db_user"]
        self.password = conf["DB"]["db_password"]
        self.schema_name = conf["DB"]["schema_name"]
        self.psql_cmd_path = conf["DB"]["psql_cmd_path"]

        return conf, valid

    def self2dict(self):
        return self.__dict__
