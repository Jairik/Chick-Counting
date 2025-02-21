# Database Platform Research

## What could our database choice potentially support?
  -	image metadata
    -	images generated from RGB and Raspberry Pi thermal cameras
    -	AI detections from YOLO11
  -	real-time processing
    -	quick storage and retrieval of data (ex: number of chicks detected so far)
  -	optimal scalability
  -	accurate tracking and analysis
    -	tracking and comparing AI detections to human counts


## Database Comparisons
| Database        | Strengths          | Weaknesses             |
| --------------- | ------------------ | ---------------------- |
| PostGreSQL      | Optimal ACID compliance, can define custom data types and plugins | More complex for configurations, more learning required
| MySQL | Easier to set up, less complex, good scalability/flexibility, most familiar with | Less flexible and not as powerful for AI data, may face challenges handling very large datasets, may need additional protection from data corruption
| MongoDB | Optimal image metadata storage with JSON, YOLO11 can output JSON-based data, can manage large data with lower latencies | Higher memory and resource consumption, 16mb limit on document sizes
| Firebase | Optimal real-time tracking, optimal scalability, pretty familiar with | Closed source, costly, slower querying, 1mb limit on document sizes
| AWS Dynamo DB | Optimal scalability, built for large-scale AI processing, additional support of AWS cloud, optimal real-time processing | Costly, lack of server-side updates which makes it difficult to bulk change records

## Potential Attributes Stored
- each tray (ex: tray_id)
  - for keeping track of chick counts by YOLO11
- chick counts for each tray by YOLO11
- actual chick counts for sample trays
  - can be used to keep track of the accuracy of certain experiment
- each experiment (ex: experiment_id)
  - if we need to keep track of what certain aspects of it work and which don't
- each date chicks were tracked
