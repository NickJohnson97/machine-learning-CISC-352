(define (problem p3-dungeon)
  (:domain Dungeon)
  (:objects
            cell11 cell12 cell13 cell14 cell15 cell21 cell22 cell23 cell24 cell25  cell31 cell32 cell33 cell34 cell35 cell41 cell42 cell43 cell44 cell45 - cells
            sword1 sword2 sword3 sword4 - swords
  )
  (:init

    ;Initial Hero Location
    (at-hero cell45)
    
    ;Hero starts with a free arm
    (arm-free)

    ;Initial location of the swords
    (at-sword sword1 cell15)
    (at-sword sword2 cell23)
    (at-sword sword3 cell31)
    (at-sword sword4 cell43)
    
    ;Initial location of Monsters
    (has-monster cell13)
    (has-monster cell14)
    (has-monster cell22)
    (has-monster cell32)
    (has-monster cell34)
    (has-monster cell42)
    (has-monster cell44)
    
    ;Initial lcocation of Traps
    (has-trap cell12)
    (has-trap cell21)
    (has-trap cell24)
    (has-trap cell25)
    (has-trap cell33)
    (has-trap cell41)
    
    ;Graph Connectivity
    ;left out connection to the staring cell
    ;left out connection from the goal cell
    ;these moves should never occur anyway
    (connected cell12 cell11)
    (connected cell12 cell22)
    (connected cell12 cell13)

    (connected cell13 cell12)
    (connected cell13 cell23)
    (connected cell13 cell14)

    (connected cell14 cell13)
    (connected cell14 cell24)
    (connected cell14 cell15)

    (connected cell15 cell14)
    (connected cell15 cell25)

    (connected cell21 cell11)
    (connected cell21 cell22)
    (connected cell21 cell31)

    (connected cell22 cell12)
    (connected cell22 cell21)
    (connected cell22 cell32)
    (connected cell22 cell23)

    (connected cell23 cell13)
    (connected cell23 cell22)
    (connected cell23 cell33)
    (connected cell23 cell24)

    (connected cell24 cell14)
    (connected cell24 cell23)
    (connected cell24 cell34)
    (connected cell24 cell25)

    (connected cell25 cell15)
    (connected cell25 cell24)
    (connected cell25 cell35)

    (connected cell31 cell21)
    (connected cell31 cell32)
    (connected cell31 cell41)

    (connected cell32 cell22)
    (connected cell32 cell31)
    (connected cell32 cell42)
    (connected cell32 cell33)

    (connected cell33 cell23)
    (connected cell33 cell32)
    (connected cell33 cell43)
    (connected cell33 cell34)

    (connected cell34 cell24)
    (connected cell34 cell33)
    (connected cell34 cell44)
    (connected cell34 cell35)

    (connected cell35 cell25)
    (connected cell35 cell34)

    (connected cell41 cell31)
    (connected cell41 cell42)

    (connected cell42 cell41)
    (connected cell42 cell32)
    (connected cell42 cell43)

    (connected cell43 cell42)
    (connected cell43 cell33)
    (connected cell43 cell44)

    (connected cell44 cell43)
    (connected cell44 cell34)

    (connected cell45 cell44)
    (connected cell45 cell35)
  )
  (:goal (and
            ;Hero's Goal Location
              (at-hero cell11)
  ))
  
)
