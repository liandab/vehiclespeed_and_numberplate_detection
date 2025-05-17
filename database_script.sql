CREATE DATABASE vsnp;
USE vsnp;

CREATE TABLE Vehicles (
    id INT AUTO_INCREMENT PRIMARY KEY,
    number_plate VARCHAR(20),
    speed_kph INT,
    location VARCHAR(100),
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE Violations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    number_plate VARCHAR(20),
    speed_kph INT,
    speed_limit INT,
    location VARCHAR(100),
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    snapshot VARCHAR(255)
);

CREATE TABLE Users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL
);
ALTER TABLE users ADD COLUMN password VARCHAR(255) NOT NULL;


select * from vehicles;
select * from violations;
select * from users;

truncate table vehicles;
truncate table violations;
truncate table users;


truncate table users;


