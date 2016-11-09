

def enum(**args):
    enums = {}
    enums["keys"] = args.keys
    enums["values"] = args.values
    enums["names"] = args
    enums["reverse"] = dict((value, key) for key, value in args.iteritems())
    enums.update(args)
    return type('Enum', (), enums)

