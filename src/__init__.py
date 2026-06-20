"""Class Thread model — a dual-node knowledge representation prototype.

Every concept is represented by two nodes:
  * an abstract node     (the intension / idea of the concept)
  * an extension node    (the set of all instances of the concept)

linked by a ``hasExtension`` relation. A **Class Thread** is a directed path:

    hasExtension -> (zero or more subClassOf) -> instanceOf

connecting a concept's abstract node to a concrete instance.
"""

__version__ = "0.1.0"
