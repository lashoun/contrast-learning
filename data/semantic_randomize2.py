"""
SemanticRandomize -- simulate clusters by adding noise to archetype points
first author: Jean-Louis Dessalles (2017), major refactoring: Giovanni Sileno (2018)
"""

import sys
import os
import csv
import numpy as np
import random
import helpers as h

###################################
# script parameters and functions #
###################################

DEFAULT_CLUSTER_SIZE = 20  # number of objects per generated cluster
DEFAULT_OUTPUT_FILE_SUFFIX = "scattered"

def usage():
    cmd = os.path.basename(sys.argv[0])
    print('''Usage:

$ python %s <cards_suit_archetypes.csv> [<cluster_size>] [<samples.csv>]

The program generates <cluster_size> (default: 20) instances of the prototypes specified 
in <cards_suit_archetypes.csv> and records them in <samples.csv> (default: <archetypes_scattered.csv>).

The file <cards_suit_archetypes.csv> contains a table with variations or values.
The first line is 'name' and other feature names; then one prototype is specified for each line.
Variations can be 
 - intervals: "17..34", "orange..yellow", "(13, 35)..(16, 36)" 
 - lists: "17, 18, 19", "orange, blue, green", "(13, 35), (16, 36)"
 - normal draws: "N(3, 1)", where 3 is mean, 1 stdev''' % cmd)

######################
# computational core #
######################

class Instance:
    """Instances have name (id) and coordinates"""

    def __init__(self, name, coordinates):
        self.name = name
        self.coordinates = coordinates

    def to_row(self):
        return tuple([self.name] + [str(c) for c in self.coordinates])

    def __str__(self):
        return "%s;%s" % (self.name, ';'.join([str(c) for c in self.coordinates]))


class Archetype:
    """Archetypes have a name and a list of variations associated to each dimension,
       that is, parameters defining how to generate instances"""

    def __init__(self, name, variations):
        self.name = name
        self.n_instances = 0
        self.variations = variations

    def generate(self):

        def scatter(variation):
            # interval like "3..17"
            match = h.match_range_integers(variation)
            if match is not False:
                return random.randint(match[0], match[1])

            # interval like "orange..yellow"
            match = h.match_range_colours(variation)
            if match is not False:
                n = len(match[0]) # extract the number of dimensions
                assert n == 3     # we expect 3 dimensions for colors
                output = []
                for i in range(0, n):
                    if match[0][i] <= match[1][i]:
                        output.append(random.randint(match[0][i], match[1][i]))
                    else:
                        output.append(random.randint(match[1][i], match[0][i]))
                return tuple(output)

            # interval like "(25, 25)..(45, 62)"
            match = h.match_range_tuples(variation)
            if match is not False:
                n = len(match[0])  # extract the number of dimensions
                output = []
                for i in range(0, n):
                    if match[0][i] <= match[1][i]:
                        output.append(random.randint(match[0][i], match[1][i]))
                    else:
                        output.append(random.randint(match[1][i], match[0][i]))
                return tuple(output)

            # normal distribution template like "N(102, 11)" where 102 = mean, 11 = std dev
            match = h.match_normal_template(variation)
            if match is not False:
                return round(np.random.normal(match[0], match[1]), 3)

            # single integer
            match = h.match_integer(variation)
            if match is not False:
                return match

            # single float
            match = h.match_float(variation)
            if match is not False:
                return match

            # single color like "yellow", return (255, 255, 0)
            match = h.match_colour(variation)
            if match is not False:
                return match

            # single tuple like "(255, 255, 0, 2, 5)"
            match = h.match_tuple(variation)
            if match is not False:
                return match

            # single string like "dog"
            match = h.match_string(variation)
            if match is not False:
                return match

            # list of colors
            match = h.match_list_colours(variation)
            if match is not False:
                return random.choice(match)

            # list of strings
            match = h.match_list_strings(variation)
            if match is not False:
                return random.choice(match)

            # list of integers
            match = h.match_list_integers(variation)
            if match is not False:
                return random.choice(match)

            raise ValueError("Unknown variation: "+ variation)

        # increase the cardinality (and use it as a internal key for objects)
        self.n_instances += 1
        # return the object, with name given by the class + id,
        # and coordinates given as a conjunction of coordinates generated by the class variations
        return Instance('%s_%d' % (self.name, self.n_instances), list(map(scatter, self.variations)))


class ArchetypeSet():

    def __init__(self, csv_filename):
        # read csv file in memory, separating the header
        with open(csv_filename, newline='') as f:
            reader = csv.reader(f, delimiter=';', quotechar='"')
            rows = [m for m in reader]

        header = rows.pop(0)
        if header[0].lower() != 'name':
            raise ValueError("The first field in the first row has to be 'name' or 'Name'.")

        self.dimensions = header[1:]

        # open up the data
        content = np.array(rows[:])

        # extract the element names
        self.names = content[:, 0]

        # extract the values, expanding the tuple columns
        self.variations = content[:, 1:]

    def generate_rows(self, cluster_size):
        # generate samples for each archetype
        samples = []
        for i in range(0, len(self.variations)):
            name = self.names[i] # the first column gives the name of the class
            variations = self.variations[i]
            archetype = Archetype(name, variations)
            for i in range(0, cluster_size):
                instance = archetype.generate()
                samples.append(instance.to_row())
        return samples

    def generate_csv(self, output_filename, cluster_size):

        # generate instances
        samples = self.generate_rows(cluster_size)

        # organize the rows for the output table
        output_rows = []
        output_rows.append(["name"] + self.dimensions)
        for sample in samples:
            output_rows.append(sample)

        # write the output as csv
        with open(output_filename, 'w') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerows(output_rows)


##########
# script #
##########

if __name__ == "__main__":
    print(__doc__)

    # read arguments and define behaviour
    if 1 < len(sys.argv) < 4:
        archetypes_filename = sys.argv.pop(1)

        if len(sys.argv) > 1 and sys.argv[1].isdigit():
            cluster_size = int(sys.argv.pop(1))
        else:
            cluster_size = DEFAULT_CLUSTER_SIZE

        if len(sys.argv) > 1:
            output_filename = sys.argv[2]
        else:
            output_filename = os.path.splitext(archetypes_filename)[0] + "_" + DEFAULT_OUTPUT_FILE_SUFFIX + '.csv'
    else:
        usage()
        exit(1)

    archetype_set = ArchetypeSet(archetypes_filename)
    archetype_set.generate_csv(output_filename, cluster_size)
    print('%d x %d = %d objects generated' % (cluster_size, len(archetype_set.names), cluster_size * len(archetype_set.names)))
    print('recorded in %s' % output_filename)
