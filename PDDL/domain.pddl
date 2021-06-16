(define (domain Dungeon)

    (:requirements
        :typing
        :negative-preconditions
    )

    (:types
        swords cells
    )

    (:predicates
        ;Hero's cell location
        (at-hero ?loc - cells)
        
        ;Sword cell location
        (at-sword ?s - swords ?loc - cells)
        
        ;Indicates if a cell location has a monster
        (has-monster ?loc - cells)
        
        ;Indicates if a cell location has a trap
        (has-trap ?loc - cells)
        
        ;Indicates if a cell or sword has been destroyed
        (is-destroyed ?obj)
        
        ;connects cells
        ;we intentionally left out connection from the goal state since leaving is impossible
        (connected ?from ?to - cells)
        
        ;Hero's hand is free
        (arm-free)
        
        ;Hero's holding a sword
        (holding ?s - swords)
    
        ;It becomes true when a trap is disarmed
        (trap-disarmed ?loc)
        
    )

    ;Hero can move if the
    ;    - hero is at current location
    ;    - cells are connected, 
    ;    - there is no trap in current loc, and 
    ;    - destination does not have a trap/monster/has-been-destroyed
    ;Effects move the hero, and destroy the original cell. No need to destroy the sword.
    (:action move
        :parameters (?from ?to - cells)
        :precondition (and 
                            (not (has-trap ?from))
                            (not (has-monster ?to))
                            (not (has-trap ?to))
                            (not (is-destroyed ?to))
                            (at-hero ?from)
                            (connected ?from ?to)
        )
        :effect (and 
                            ;making sure we dont duplicate the hero by leaving a copy in a previous room
                            (not (at-hero ?from))
                            (at-hero ?to)
                            (is-destroyed ?from)
                )
    )
    
    ;When this action is executed, the hero gets into a location with a trap
    (:action move-to-trap
        :parameters (?from ?to - cells)
        :precondition (and 
                            (not (has-trap ?from))
                            (has-trap ?to)
                            (arm-free)
                            (at-hero ?from)
                            (connected ?from ?to)
        )
        :effect (and 
                            (not (at-hero ?from))
                            (at-hero ?to)
                            (is-destroyed ?from)
                )
    )

    ;When this action is executed, the hero gets into a location with a monster
    (:action move-to-monster
        :parameters (?from ?to - cells ?s - swords)
        :precondition (and 
                            ;checking whether the previous room had a trap
                            ;moving from a trap room to a monster room is a logical impossibility
                            (not (has-trap ?from))
                            (has-monster ?to)
                            (holding ?s)
                            (at-hero ?from)
                            (connected ?from ?to)
        )
        :effect (and 
                            (not (at-hero ?from))
                            (at-hero ?to)
                            (is-destroyed ?from)
                )
    )
    
    ;Hero picks a sword if he's in the same location
    (:action pick-sword
        :parameters (?loc - cells ?s - swords)
        :precondition (and 
                            (arm-free)
                            (at-sword ?s ?loc)
                            (at-hero ?loc)
                      )
        :effect (and
                            (holding ?s)
                            (not (arm-free))
                )
    )
    
    ;Hero destroys his sword. 
    (:action destroy-sword
        :parameters (?loc - cells ?s - swords)
        :precondition (and 
                            ;not allowing the sword to be destroyed if there is a monster
                            ;this would never be an optimal move
                            (not (has-monster ?loc))
                            (not (has-trap ?loc))
                            (holding ?s)
                            (at-hero ?loc)
                      )
        :effect (and
                            (arm-free)
                            (not (holding ?s))
                            (is-destroyed ?s)
                )
    )
    
    ;Hero disarms the trap with his free arm
    (:action disarm-trap
        :parameters (?loc - cells)
        :precondition (and 
                            (has-trap ?loc)
                            (arm-free)
                            (at-hero ?loc)
                      )
        :effect (and
                            (trap-disarmed ?loc)
                            (not (has-trap ?loc))
                )
    )
    
)