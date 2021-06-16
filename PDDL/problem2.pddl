(define (problem p2-dungeon)
  (:domain Dungeon)
  (:objects
            cell1 cell2 cell3 cell4 cell5 cell6 cell7 cell8 cell9 cell10 cell11 cell12 - cells
            sword1 sword2 - swords
  )
  (:init
  
    ;Initial Hero Location
    (at-hero cell7)
    
    ;Hero starts with a free arm
    (arm-free)
    
    ;Initial location of the swords
    (at-sword sword1 cell6)
    (at-sword sword2 cell10)
    
    ;Initial location of Monsters
    (has-monster cell5)
    (has-monster cell4)
    (has-monster cell9)
    
    ;Initial lcocation of Traps
    (has-trap cell2)
    (has-trap cell3)
    (has-trap cell8)
    (has-trap cell11)
    (has-trap cell12)
    
    ;Graph Connectivity
    ;left out connection to the staring cell
    ;left out connection from the goal cell
    ;these moves should never occur anyway
    (connected cell7 cell6)
    (connected cell7 cell12)

    (connected cell6 cell5)
    (connected cell6 cell11)

    (connected cell12 cell5)
    (connected cell12 cell11)
    
    (connected cell5 cell6)
    (connected cell5 cell12)
    (connected cell5 cell10)
    (connected cell5 cell4)

    (connected cell11 cell12)
    (connected cell11 cell6)
    (connected cell11 cell10)
    (connected cell11 cell4)

    (connected cell4 cell5)
    (connected cell4 cell11)
    (connected cell4 cell9)
    (connected cell4 cell3)

    (connected cell10 cell5)
    (connected cell10 cell11)
    (connected cell10 cell9)
    (connected cell10 cell3)

    (connected cell3 cell4)
    (connected cell3 cell10)
    (connected cell3 cell8)
    (connected cell3 cell2)

    (connected cell9 cell4)
    (connected cell9 cell10)
    (connected cell9 cell8)
    (connected cell9 cell2)

    (connected cell2 cell1)
    (connected cell2 cell3)
    (connected cell2 cell9)

    (connected cell8 cell1)
    (connected cell8 cell3)
    (connected cell8 cell9)
  )
  (:goal (and
            ;Hero's Goal Location
            (at-hero cell1)
  ))
  
)
