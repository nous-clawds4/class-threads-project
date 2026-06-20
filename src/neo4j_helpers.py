"""Optional Neo4j + Cypher helpers for the Class Thread model.

This module is intentionally a *stub* in the first version. Neo4j is NOT a
dependency of the core prototype (see requirements.txt). The functions below
sketch how the in-memory NetworkX model would map onto Cypher so that hot
paths can later be migrated without rethinking the data model.

Nothing here is imported by the core pipeline; import it explicitly only when
you have a running Neo4j instance and have installed the optional `neo4j`
driver.
"""

from __future__ import annotations

# A Class Thread in Cypher terms (the "Downhill forward path" topology):
#   (a:Abstract)-[:hasExtension]->(:Extension)
#               -[:supersetOf*0..]->(:Extension)   # general -> specific
#               -[:hasElement]->(i:Instance)
#
# The exact translation is filled in once the NetworkX model is finalized.

CLASS_THREAD_CYPHER = """
// Find all valid Class Threads reaching instances of a concept.
// Pattern: hasExtension -> (supersetOf*) -> hasElement
MATCH (a:Abstract {name: $concept})-[:hasExtension]->(:Extension)
      -[:supersetOf*0..]->(spec:Extension)-[:hasElement]->(i:Instance)
RETURN a.name AS concept, i.name AS instance
"""


def export_to_cypher(graph) -> str:  # pragma: no cover - stub
    """Serialize a NetworkX Class Thread graph to Cypher CREATE statements.

    Not implemented in the first version. Raises to make the boundary explicit.
    """
    raise NotImplementedError(
        "Neo4j export is an optional future module; the core demo uses NetworkX."
    )
